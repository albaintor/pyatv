"""Tests for experimental Companion Siri voice helpers."""

import struct
from unittest.mock import AsyncMock, Mock, call

import pytest

from pyatv import exceptions
from pyatv.protocols.companion.api import CompanionAPI, HidCommand
from pyatv.protocols.companion.voice import (
    VoiceFrameInjector,
    _opus_packets_from_ogg,
    iter_voice_content,
    voice_content,
)


def _ogg_page(packets):
    body = b"".join(packets)
    segments = bytes(len(packet) for packet in packets)
    return (
        b"OggS"
        + bytes([0, 0])
        + struct.pack("<QIIIB", 0, 1, 0, 0, len(segments))
        + segments
        + body
    )


def test_extract_opus_packets_from_ogg():
    data = _ogg_page([b"OpusHead-data", b"OpusTags-data", b"\x48voice"])

    assert _opus_packets_from_ogg(data) == [b"\x48voice"]


def test_extract_opus_packets_rejects_invalid_stream():
    with pytest.raises(exceptions.ProtocolError):
        _opus_packets_from_ogg(b"not ogg")


def test_voice_content_contains_offsets_and_sizes():
    content = voice_content([b"abc", b"de"], 0.25)

    assert content == {3: b"abcde", 4: 0.25, 5: [{1: 3, 2: 0}, {1: 2, 2: 3}]}


def test_iter_voice_content_batches_frames():
    messages = list(iter_voice_content([b"a", b"bb", b"ccc"], 2))

    assert messages == [
        {3: b"abb", 4: 0.0, 5: [{1: 1, 2: 0}, {1: 2, 2: 1}]},
        {3: b"ccc", 4: 0.04, 5: [{1: 3, 2: 0}]},
    ]


def test_voice_frame_injector_uses_original_batch_size_and_timing():
    injector = VoiceFrameInjector([b"one", b"two", b"three"])
    injector.reset()

    assert injector.replace({4: 1.5, 5: [{}, {}]}) == {
        3: b"onetwo",
        4: 1.5,
        5: [{1: 3, 2: 0}, {1: 3, 2: 3}],
    }
    assert injector.replace({4: 2.0, 5: [{}]}) == {
        3: b"three",
        4: 2.0,
        5: [{1: 5, 2: 0}],
    }
    assert injector.replace({4: 2.5, 5: [{}]}) is None


@pytest.mark.asyncio
async def test_siri_releases_button_when_audio_send_fails():
    core = Mock()
    core.loop.time.return_value = 0.0
    api = CompanionAPI(core)
    api.hid_command = AsyncMock()
    api._send_command = AsyncMock(return_value={})
    api._send_event = AsyncMock(side_effect=exceptions.ProtocolError("failed"))

    with pytest.raises(exceptions.ProtocolError):
        await api.siri([b"voice"])

    assert api._send_command.await_args_list == [
        call("_siriStart", {}),
        call("_siriStop", {}),
    ]
    assert api.hid_command.await_args_list == [
        call(True, HidCommand.Siri),
        call(False, HidCommand.Siri),
    ]

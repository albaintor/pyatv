"""Experimental helpers for Companion Siri voice input."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pyatv import exceptions

SAMPLE_RATE = 16000
FRAME_DURATION = 0.020
FRAMES_PER_MESSAGE = 5


def _opus_packets_from_ogg(data: bytes) -> List[bytes]:
    """Extract packets from an Ogg stream, including packets spanning pages."""
    packets: List[bytes] = []
    packet = bytearray()
    offset = 0

    while offset < len(data):
        if len(data) - offset < 27 or data[offset : offset + 4] != b"OggS":
            raise exceptions.ProtocolError("invalid Ogg Opus stream")

        segment_count = data[offset + 26]
        header_end = offset + 27 + segment_count
        if header_end > len(data):
            raise exceptions.ProtocolError("truncated Ogg segment table")

        segments = data[offset + 27 : header_end]
        page_end = header_end + sum(segments)
        if page_end > len(data):
            raise exceptions.ProtocolError("truncated Ogg page")

        body_offset = header_end
        for segment_size in segments:
            packet.extend(data[body_offset : body_offset + segment_size])
            body_offset += segment_size
            if segment_size < 255:
                packets.append(bytes(packet))
                packet.clear()

        offset = page_end

    if packet:
        raise exceptions.ProtocolError("truncated Ogg packet")
    if (
        len(packets) < 3
        or packets[0][:8] != b"OpusHead"
        or packets[1][:8] != b"OpusTags"
    ):
        raise exceptions.ProtocolError("missing Ogg Opus headers")
    return packets[2:]


async def encode_audio_file(filename: str) -> List[bytes]:
    """Encode an audio file as 16 kHz mono, 20 ms Opus packets using ffmpeg."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        filename,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "libopus",
        "-application",
        "voip",
        "-frame_duration",
        "20",
        "-b:a",
        "16k",
        "-f",
        "opus",
        "pipe:1",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as ex:
        raise exceptions.NotSupportedError(
            "ffmpeg with libopus support is required for Companion Siri"
        ) from ex

    stdout, stderr = await process.communicate()
    if process.returncode:
        error = stderr.decode(errors="replace").strip()
        raise exceptions.ProtocolError(f"failed to encode Siri audio: {error}")

    packets = _opus_packets_from_ogg(stdout)
    if not packets:
        raise exceptions.ProtocolError("audio file did not contain any audio packets")
    return packets


def voice_content(frames: Sequence[bytes], timestamp: float) -> Dict[int, object]:
    """Build the observed Companion ``_siA`` content dictionary."""
    offset = 0
    descriptions = []
    for frame in frames:
        descriptions.append({1: len(frame), 2: offset})
        offset += len(frame)
    return {
        3: b"".join(frames),
        4: float(timestamp),
        5: descriptions,
    }


def iter_voice_content(
    frames: Sequence[bytes], frames_per_message: int = FRAMES_PER_MESSAGE
) -> Iterable[Dict[int, object]]:
    """Yield ``_siA`` content dictionaries for a sequence of Opus packets."""
    if frames_per_message < 1:
        raise ValueError("frames_per_message must be at least one")
    for index in range(0, len(frames), frames_per_message):
        yield voice_content(
            frames[index : index + frames_per_message], index * FRAME_DURATION
        )


@dataclass
class VoiceFrameInjector:
    """Replace proxied microphone frames with frames from an audio file."""

    frames: Sequence[bytes]
    index: int = 0
    started_at: float = 0.0

    def reset(self) -> None:
        """Start injecting from the beginning of the audio clip."""
        self.index = 0
        self.started_at = time.monotonic()

    @property
    def finished(self) -> bool:
        """Return whether every replacement frame has been consumed."""
        return self.index >= len(self.frames)

    def replace(self, original: Mapping[int, Any]) -> Optional[Dict[int, object]]:
        """Replace one captured ``_siA`` payload, preserving its timing field."""
        if self.finished:
            return None

        descriptions = original.get(5, [])
        frame_count = max(1, len(descriptions))
        replacement = self.frames[self.index : self.index + frame_count]
        self.index += len(replacement)
        timestamp = original.get(4, time.monotonic() - self.started_at)
        return voice_content(replacement, float(timestamp))

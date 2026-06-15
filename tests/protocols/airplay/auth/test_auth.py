"""Unit tests for pyatv.protocols.airplay.auth."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyatv import exceptions
from pyatv.auth import hap_tlv8
from pyatv.auth.hap_pairing import NO_CREDENTIALS, HapCredentials
from pyatv.protocols.airplay.auth import (
    AuthenticationType,
    HapCredentials,
    NullPairVerifyProcedure,
    pair_setup,
    pair_verify,
)
from pyatv.protocols.airplay.auth.hap import (
    AirPlayHapPairSetupProcedure,
    AirPlayHapPairVerifyProcedure,
)
from pyatv.protocols.airplay.auth.legacy import (
    AirPlayLegacyPairSetupProcedure,
    AirPlayLegacyPairVerifyProcedure,
)
from pyatv.support.http import HttpResponse

# Legacy credentials only have ltsk (seed) and client_id (identifier) filled in
LEGACY_CREDENTIALS = HapCredentials(b"", b"1", b"", b"2")

HAP_CREDENTIALS = HapCredentials(b"1", b"2", b"3", b"4")


@pytest.fixture(name="srp")
def srp_fixture():
    with patch("pyatv.protocols.airplay.auth.LegacySRPAuthHandler") as srp:
        yield srp


@pytest.fixture(name="connection")
def connection_fixture():
    yield MagicMock()


# No authentication


def test_pair_verify_no_credentials(srp, connection):
    procedure = pair_verify(NO_CREDENTIALS, connection)

    srp.assert_not_called()
    assert isinstance(procedure, NullPairVerifyProcedure)


# Legacy authentication


@patch("pyatv.protocols.airplay.auth.new_credentials", return_value=LEGACY_CREDENTIALS)
def test_pair_setup_legacy(new_credentials, srp, connection):
    procedure = pair_setup(AuthenticationType.Legacy, connection)

    srp.assert_called_with(LEGACY_CREDENTIALS)
    assert isinstance(procedure, AirPlayLegacyPairSetupProcedure)


def test_pair_verify_legacy(srp, connection):
    procedure = pair_verify(LEGACY_CREDENTIALS, connection)

    srp.assert_called_with(LEGACY_CREDENTIALS)
    assert isinstance(procedure, AirPlayLegacyPairVerifyProcedure)


# HAP authentication


def test_pair_setup_hap(connection):
    procedure = pair_setup(AuthenticationType.HAP, connection)
    assert isinstance(procedure, AirPlayHapPairSetupProcedure)


def test_pair_verify_hap(connection):
    procedure = pair_verify(HAP_CREDENTIALS, connection)
    assert isinstance(procedure, AirPlayHapPairVerifyProcedure)


@pytest.mark.asyncio
async def test_pair_setup_hap_uses_pair_setup():
    connection = MagicMock()
    connection.post = AsyncMock(
        side_effect=[
            HttpResponse("HTTP", "1.1", 200, "OK", {}, b""),
            HttpResponse(
                "HTTP",
                "1.1",
                200,
                "OK",
                {},
                hap_tlv8.write_tlv(
                    {
                        hap_tlv8.TlvValue.SeqNo: b"\x02",
                        hap_tlv8.TlvValue.Salt: b"salt",
                        hap_tlv8.TlvValue.PublicKey: b"public_key",
                    }
                ),
            ),
        ]
    )
    srp = MagicMock()

    with patch("pyatv.protocols.airplay.auth.hap.asyncio.sleep", new=AsyncMock()):
        await AirPlayHapPairSetupProcedure(connection, srp).start_pairing()

    pairing_body = connection.post.call_args_list[1].kwargs["body"]
    pairing_data = hap_tlv8.read_tlv(pairing_body)
    assert pairing_data[hap_tlv8.TlvValue.Method] == int.to_bytes(
        hap_tlv8.Method.PairSetup.value, 1, byteorder="big"
    )
    assert pairing_data[hap_tlv8.TlvValue.SeqNo] == b"\x01"


@pytest.mark.asyncio
async def test_pair_setup_hap_starts_pin_window_without_body():
    connection = MagicMock()
    connection.post = AsyncMock(
        side_effect=[
            HttpResponse("HTTP", "1.1", 200, "OK", {}, b""),
            HttpResponse(
                "HTTP",
                "1.1",
                200,
                "OK",
                {},
                hap_tlv8.write_tlv(
                    {
                        hap_tlv8.TlvValue.SeqNo: b"\x02",
                        hap_tlv8.TlvValue.Salt: b"salt",
                        hap_tlv8.TlvValue.PublicKey: b"public_key",
                    }
                ),
            ),
        ]
    )
    srp = MagicMock()

    sleep = AsyncMock()
    with patch("pyatv.protocols.airplay.auth.hap.asyncio.sleep", new=sleep):
        await AirPlayHapPairSetupProcedure(connection, srp).start_pairing()

    assert connection.post.call_args_list[0].args == ("/pair-pin-start",)
    assert connection.post.call_args_list[0].kwargs == {
        "headers": {
            "User-Agent": "AirPlay/320.20",
            "Connection": "keep-alive",
            "X-Apple-HKP": 3,
            "Content-Length": "0",
        }
    }
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_pair_setup_hap_raises_on_backoff():
    connection = MagicMock()
    connection.post = AsyncMock(
        side_effect=[
            HttpResponse("HTTP", "1.1", 200, "OK", {}, b""),
            HttpResponse(
                "HTTP",
                "1.1",
                200,
                "OK",
                {},
                hap_tlv8.write_tlv(
                    {
                        hap_tlv8.TlvValue.SeqNo: b"\x02",
                        hap_tlv8.TlvValue.Error: bytes([hap_tlv8.ErrorCode.BackOff]),
                        hap_tlv8.TlvValue.BackOff: b"\xae",
                    }
                ),
            ),
        ]
    )
    srp = MagicMock()

    with patch("pyatv.protocols.airplay.auth.hap.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(exceptions.BackOffError, match="174 seconds"):
            await AirPlayHapPairSetupProcedure(connection, srp).start_pairing()

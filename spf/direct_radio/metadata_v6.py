"""ABI 3 metadata: explicit tandem ownership or legacy gain observations.

The fixed extension adds temperature and exact gap accounting to the existing
validated v3/v4 records. Legacy records never claim tandem events or a lease.
"""

from __future__ import annotations

import dataclasses
import struct
import zlib

from spf.direct_radio.tandem_agc import (
    HEADER_PREFIX_BYTES_V4,
    TANDEM_METADATA_FEATURE,
    TANDEM_METADATA_VALID_FLAG,
    RadioMetadataV4,
)
from spf.direct_radio.usb_protocol import (
    HEADER_PREFIX_BYTES_V3,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    RadioMetadataV3,
)

_TEMPERATURE = 1 << 9
_LAYOUT = 1 << 10
_EXACT_GAPS = 1 << 11
_GAP_FLAG = 1 << 23
_TANDEM_FEATURES = 0xFFF
_LEGACY_FEATURES = _TANDEM_FEATURES & ~(TANDEM_METADATA_FEATURE | 8)


def _seal(raw: bytearray) -> bytes:
    raw[-4:] = bytes(4)
    struct.pack_into("<I", raw, len(raw) - 4, zlib.crc32(raw))
    return bytes(raw)


@dataclasses.dataclass(frozen=True)
class RadioMetadataV6:
    base: RadioMetadataV3
    tandem: RadioMetadataV4 | None
    header_bytes: int
    missing_samples_before: int
    ad9361_temperature_mdeg_c: int | None
    features: MetadataFeatures
    flags: MetadataFlags

    def __getattr__(self, name):
        return getattr(self.tandem if self.tandem is not None else self.base, name)

    @property
    def ownership_epoch(self):
        return self.tandem.ownership_epoch if self.tandem else None

    @classmethod
    def unpack(cls, header):
        raw = bytes(header)
        if len(raw) < HEADER_PREFIX_BYTES_V4 + 4:
            raise ProtocolError("short protocol v6 metadata header")
        magic, version, size, features, flags = struct.unpack_from("<IHHII", raw)
        if magic != 0x314D4753 or version != 6 or size != len(raw):
            raise ProtocolError("bad protocol v6 identity or length")
        if raw != _seal(bytearray(raw)):
            raise ProtocolError("protocol v6 metadata CRC mismatch")
        if features not in (_TANDEM_FEATURES, _LEGACY_FEATURES):
            raise ProtocolError("unknown protocol v6 feature set")
        is_tandem = features == _TANDEM_FEATURES
        if (
            flags & ~((1 << 24) - 1)
            or bool(flags & TANDEM_METADATA_VALID_FLAG) != is_tandem
        ):
            raise ProtocolError("protocol v6 ownership flags disagree")
        samples, iq_bytes, mask = struct.unpack_from("<III", raw, 40)
        if not samples or iq_bytes != samples * 8 or mask != 15 or raw[54] != 2:
            raise ProtocolError("SPF metadata requires canonical dual-RX IQ")
        missing = struct.unpack_from("<Q", raw, 116)[0]
        if bool(flags & _GAP_FLAG) != bool(missing):
            raise ProtocolError("protocol v6 gap count and flag disagree")
        temperature = struct.unpack_from("<i", raw, HEADER_PREFIX_BYTES_V3 + 40)[0]
        if any(raw[HEADER_PREFIX_BYTES_V3 + 44 : HEADER_PREFIX_BYTES_V4]):
            raise ProtocolError("protocol v6 reserved fields are nonzero")

        normalized = bytearray(raw)
        struct.pack_into("<II", normalized, 116, 0, 0)
        struct.pack_into("<I", normalized, 12, flags & ~_GAP_FLAG)
        if is_tandem:
            struct.pack_into("<H", normalized, 4, 4)
            struct.pack_into(
                "<I", normalized, 8, features & ~(_TEMPERATURE | _LAYOUT | _EXACT_GAPS)
            )
            struct.pack_into("<i", normalized, HEADER_PREFIX_BYTES_V3 + 40, 0)
            tandem = RadioMetadataV4.unpack(_seal(normalized))
            base = tandem.base
        else:
            if any(raw[HEADER_PREFIX_BYTES_V3 : HEADER_PREFIX_BYTES_V3 + 40]):
                raise ProtocolError("legacy metadata unexpectedly claims tandem state")
            if struct.unpack_from("<HH", raw, 102) != (0, 0):
                raise ProtocolError("legacy metadata unexpectedly claims FPGA events")
            del normalized[HEADER_PREFIX_BYTES_V3:HEADER_PREFIX_BYTES_V4]
            struct.pack_into("<HH", normalized, 4, 3, len(normalized))
            struct.pack_into(
                "<I", normalized, 8, features & ~(_TEMPERATURE | _LAYOUT | _EXACT_GAPS)
            )
            base = RadioMetadataV3.unpack(_seal(normalized))
            tandem = None
        return cls(
            base,
            tandem,
            size,
            missing,
            None if temperature == -(1 << 31) else temperature,
            MetadataFeatures(features),
            MetadataFlags(flags),
        )

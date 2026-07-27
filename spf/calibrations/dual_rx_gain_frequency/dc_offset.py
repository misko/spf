"""Read-only decoding of AD9361 RF DC offset correction words."""

from __future__ import annotations

from typing import Any

from spf.bench.dual_rx_phase import resolve_pluto_uri


RF_DC_REGISTER_BASE = {
    "A": 0x174,
    "B_C": 0x17D,
}
STUCK_CORRECTION_WORD = 0x200


def signed_10bit(value: int) -> int:
    """Decode one 10-bit two's-complement correction word."""

    value = int(value)
    if not 0 <= value <= 0x3FF:
        raise ValueError(f"10-bit correction word outside range: {value}")
    return value - 0x400 if value & 0x200 else value


def decode_rf_dc_correction_words(
    registers: dict[int, int],
    *,
    input_port: str = "A",
) -> dict[str, dict[str, int | bool]]:
    """Decode the four packed RF DC correction words from five registers.

    The packing follows the ADI ``AD936x_DCOFFSET_ISSUE`` register table.
    ``input_port='B_C'`` selects the correction bank shared by RF inputs B/C.
    """

    try:
        base = RF_DC_REGISTER_BASE[input_port]
    except KeyError as error:
        raise ValueError(
            f"unsupported RF input correction bank: {input_port}"
        ) from error
    try:
        r0, r1, r2, r3, r4 = (
            int(registers[address]) for address in range(base, base + 5)
        )
    except KeyError as error:
        raise ValueError(
            f"missing RF DC register 0x{int(error.args[0]):03x}"
        ) from error
    if any(not 0 <= value <= 0xFF for value in (r0, r1, r2, r3, r4)):
        raise ValueError("RF DC registers must contain bytes")

    raw = {
        "rx1_q": ((r1 & 0x03) << 8) | r0,
        "rx1_i": ((r2 & 0x0F) << 6) | (r1 >> 2),
        "rx2_q": ((r3 & 0x3F) << 4) | (r2 >> 4),
        "rx2_i": (r4 << 2) | (r3 >> 6),
    }
    return {
        name: {
            "raw": value,
            "signed": signed_10bit(value),
            "is_documented_stuck_value": value == STUCK_CORRECTION_WORD,
        }
        for name, value in raw.items()
    }


def read_rf_dc_registers(control_device: Any, *, input_port: str = "A") -> dict:
    """Read and decode one correction bank without mutating the radio."""

    try:
        base = RF_DC_REGISTER_BASE[input_port]
    except KeyError as error:
        raise ValueError(
            f"unsupported RF input correction bank: {input_port}"
        ) from error
    registers = {
        address: int(control_device.reg_read(address))
        for address in range(base, base + 5)
    }
    return {
        "input_port": input_port,
        "registers": {
            f"0x{address:03x}": value for address, value in registers.items()
        },
        "correction_words": decode_rf_dc_correction_words(
            registers,
            input_port=input_port,
        ),
    }


def inspect_radio_rf_dc(
    *,
    serial: str | None = None,
    uri: str | None = None,
    adi_module=None,
    scan_contexts=None,
) -> dict:
    """Read identity, gain/tracking state, and both correction banks."""

    resolved_uri = resolve_pluto_uri(
        uri=uri,
        serial=serial,
        scan_contexts=scan_contexts,
    )
    if adi_module is None:
        import adi as adi_module

    sdr = adi_module.ad9361(uri=resolved_uri)
    actual_serial = sdr._ctx.attrs.get("hw_serial")
    if serial is not None and actual_serial != serial:
        raise RuntimeError(
            f"IIO serial mismatch: requested {serial}, opened {actual_serial}"
        )
    channels = {}
    for channel_name in ("voltage0", "voltage1"):
        channel = sdr._ctrl.find_channel(channel_name, is_output=False)
        channels[channel_name] = {
            name: channel.attrs[name].value
            for name in (
                "gain_control_mode",
                "hardwaregain",
                "rf_dc_offset_tracking_en",
                "bb_dc_offset_tracking_en",
                "quadrature_tracking_en",
                "rf_port_select",
            )
            if name in channel.attrs
        }
    return {
        "schema": "spf.calibration.ad9361_rf_dc_register_snapshot",
        "schema_version": 1,
        "uri": resolved_uri,
        "serial": actual_serial,
        "gain_indices": {
            "rx1": int(sdr._ctrl.reg_read(0x2B0) & 0x7F),
            "rx2": int(sdr._ctrl.reg_read(0x2B5) & 0x7F),
        },
        "channels": channels,
        "correction_banks": {
            bank: read_rf_dc_registers(sdr._ctrl, input_port=bank)
            for bank in ("A", "B_C")
        },
    }

"""Fail-closed TX muting for explicitly enabled Pluto hardware tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class MutedPluto:
    serial: str
    uri: str
    tx1_gain_db: float
    tx2_gain_db: float


def validate_loopback_safety(
    *,
    physical_attenuation_db: float | None,
    strongest_tx_gain_db: float,
    minimum_effective_attenuation_db: float = 30.0,
) -> float:
    """Return worst-case effective attenuation or reject an unsafe setup.

    AD9361 TX hardware gain is non-positive, so reducing it contributes the
    corresponding positive amount of attenuation in addition to the physical
    attenuator. This permits a declared 20 dB cabled path only when every TX
    level used by the test is derated by at least another 10 dB.
    """

    if physical_attenuation_db is None:
        raise ValueError("physical loopback attenuation must be declared")
    if physical_attenuation_db < 0:
        raise ValueError("physical loopback attenuation cannot be negative")
    if not -80 <= strongest_tx_gain_db <= 0:
        raise ValueError("strongest TX gain must be between -80 and 0 dB")
    effective_attenuation_db = physical_attenuation_db - strongest_tx_gain_db
    if effective_attenuation_db < minimum_effective_attenuation_db:
        raise ValueError(
            "unsafe loopback: physical attenuation "
            f"{physical_attenuation_db:g} dB with strongest TX gain "
            f"{strongest_tx_gain_db:g} dB provides only "
            f"{effective_attenuation_db:g} dB effective attenuation; "
            f"at least {minimum_effective_attenuation_db:g} dB is required"
        )
    return effective_attenuation_db


def _usb_iio_contexts(scan_contexts) -> list[str]:
    return sorted(uri for uri in scan_contexts() if uri.startswith("usb:"))


def mute_attached_plutos(
    *,
    serials: Iterable[str] = (),
    expected_count: int | None = None,
    adi_module=None,
    scan_contexts=None,
) -> tuple[MutedPluto, ...]:
    """Mute selected USB Plutos and verify both hardware-gain readbacks.

    Every cleanup operation is attempted even if an earlier operation fails.
    This function deliberately opens only standard USB-IIO; it never claims the
    vendor streaming interface, starts RX, changes QSPI, or enables a TX path.
    """

    if adi_module is None:
        import adi as adi_module
    if scan_contexts is None:
        import iio

        scan_contexts = iio.scan_contexts

    requested = set(serials)
    discovered: set[str] = set()
    results: list[MutedPluto] = []
    failures: list[str] = []
    for uri in _usb_iio_contexts(scan_contexts):
        sdr = None
        try:
            sdr = adi_module.ad9361(uri=uri)
            serial = str(sdr._ctx.attrs.get("hw_serial", ""))
            if not serial:
                failures.append(f"{uri}: missing hw_serial")
                continue
            discovered.add(serial)
            if requested and serial not in requested:
                continue

            operation_failures = []
            operations = (
                ("mute TX1", lambda: setattr(sdr, "tx_hardwaregain_chan0", -80)),
                ("mute TX2", lambda: setattr(sdr, "tx_hardwaregain_chan1", -80)),
                ("disable DDS", sdr.disable_dds),
                (
                    "disable TX channels",
                    lambda: setattr(sdr, "tx_enabled_channels", []),
                ),
                ("destroy TX buffer", sdr.tx_destroy_buffer),
                (
                    "disable cyclic TX",
                    lambda: setattr(sdr, "tx_cyclic_buffer", False),
                ),
            )
            for name, operation in operations:
                try:
                    operation()
                except Exception as error:  # all independent safety steps run
                    operation_failures.append(
                        f"{name}: {type(error).__name__}: {error}"
                    )

            tx1 = float(sdr.tx_hardwaregain_chan0)
            tx2 = float(sdr.tx_hardwaregain_chan1)
            if tx1 > -79.75 or tx2 > -79.75:
                operation_failures.append(
                    f"mute readback mismatch: TX1={tx1} dB TX2={tx2} dB"
                )
            if operation_failures:
                failures.append(f"{serial} ({uri}): " + "; ".join(operation_failures))
                continue
            results.append(
                MutedPluto(
                    serial=serial,
                    uri=uri,
                    tx1_gain_db=tx1,
                    tx2_gain_db=tx2,
                )
            )
        except Exception as error:
            failures.append(f"{uri}: {type(error).__name__}: {error}")
        finally:
            if sdr is not None:
                # pyadi/libiio releases the context with the final reference.
                del sdr

    missing = requested - discovered
    if missing:
        failures.append(f"requested serials not found over USB-IIO: {sorted(missing)}")
    if expected_count is not None and len(results) != expected_count:
        failures.append(
            f"expected {expected_count} muted radios, verified {len(results)}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return tuple(sorted(results, key=lambda item: item.serial))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mute TX1/TX2 and DDS on attached USB Pluto radios"
    )
    parser.add_argument("--serial", action="append", default=[])
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.expected_count is not None and args.expected_count < 1:
        parser.error("--expected-count must be positive")
    muted = mute_attached_plutos(
        serials=args.serial,
        expected_count=args.expected_count,
    )
    payload = {"status": "muted", "radios": [asdict(item) for item in muted]}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

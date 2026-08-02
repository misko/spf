"""Read the passive direct-USB gadget build identity for one Pluto."""

from __future__ import annotations

import argparse

from spf.sdrpluto.direct_usb_protocol import HardwareIdentityFlags
from spf.sdrpluto.direct_usb_receiver import PlutoDirectUsbReceiver


def read_gadget_build_id(serial: str) -> str:
    with PlutoDirectUsbReceiver(serial=serial, protocol_version=2) as receiver:
        identity = receiver.query_hardware_identity()
    if not identity.flags & HardwareIdentityFlags.GADGET_BUILD_ID_VALID:
        raise RuntimeError(f"{serial}: gadget build identity is unavailable")
    return identity.gadget_build_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    args = parser.parse_args()
    try:
        build_id = read_gadget_build_id(args.serial)
    except Exception as error:
        parser.error(str(error))
    print(build_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

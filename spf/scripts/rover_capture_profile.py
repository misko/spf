"""Resolve and validate Rover 3.1 production capture profiles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1/capture_configs"


@dataclass(frozen=True)
class CaptureProfile:
    name: str
    rover_id: int
    config: str
    routine: str
    records_per_receiver: int
    expected_radios: int
    rx_transport: str
    data_version: int

    @property
    def config_path(self) -> Path:
        return CONFIG_ROOT / self.config


_LEGACY = {
    1: CaptureProfile(
        "legacy_iio_v4",
        1,
        "rover_receiver_config_pi_3mhz_35mm.yaml",
        "bounce",
        3000,
        2,
        "iio",
        4,
    ),
    2: CaptureProfile(
        "legacy_iio_v4",
        2,
        "rover_single_receiver_config_pi_3mhz.yaml",
        "circle",
        3500,
        1,
        "iio",
        4,
    ),
    3: CaptureProfile(
        "legacy_iio_v4",
        3,
        "rover_receiver_config_pi_3mhz_43mm.yaml",
        "bounce",
        3000,
        2,
        "iio",
        4,
    ),
}

_DIRECT = {
    "direct_usb_v4": CaptureProfile(
        "direct_usb_v4",
        1,
        "rover1_receiver_config_pi_3mhz_35mm_direct_usb_v2_v4.yaml",
        "bounce",
        3000,
        2,
        "direct_usb",
        4,
    ),
    "direct_usb_v7": CaptureProfile(
        "direct_usb_v7",
        1,
        "rover1_receiver_config_pi_3mhz_35mm_direct_usb_v2.yaml",
        "bounce",
        3000,
        2,
        "direct_usb",
        7,
    ),
}


def resolve_profile(name: str, rover_id: int) -> CaptureProfile:
    if name == "legacy_iio_v4":
        try:
            profile = _LEGACY[rover_id]
        except KeyError as exc:
            raise ValueError(f"unsupported rover_id: {rover_id}") from exc
    else:
        try:
            profile = _DIRECT[name]
        except KeyError as exc:
            valid = ", ".join(("legacy_iio_v4", *_DIRECT))
            raise ValueError(
                f"unknown capture profile {name!r}; choose {valid}"
            ) from exc
        if profile.rover_id != rover_id:
            raise ValueError(
                f"{name} is qualified only for Rover {profile.rover_id}, "
                f"not Rover {rover_id}"
            )
    validate_profile(profile)
    return profile


def validate_profile(profile: CaptureProfile) -> None:
    if not profile.config_path.is_file():
        raise ValueError(f"capture config does not exist: {profile.config_path}")
    with profile.config_path.open() as source:
        config = yaml.safe_load(source)
    if config.get("data-version") != profile.data_version:
        raise ValueError(
            f"{profile.config}: data-version is {config.get('data-version')}, "
            f"expected {profile.data_version}"
        )
    receivers = config.get("receivers", [])
    if len(receivers) != profile.expected_radios:
        raise ValueError(
            f"{profile.config}: has {len(receivers)} receivers, "
            f"expected {profile.expected_radios}"
        )
    transports = {receiver.get("rx-transport", "iio") for receiver in receivers}
    if transports != {profile.rx_transport}:
        raise ValueError(
            f"{profile.config}: receiver transports are {sorted(transports)}, "
            f"expected only {profile.rx_transport}"
        )
    if profile.rx_transport == "direct_usb":
        protocols = {
            receiver.get("direct-usb", {}).get("protocol-version")
            for receiver in receivers
        }
        if protocols != {2}:
            raise ValueError(
                f"{profile.config}: direct-USB protocols are {sorted(protocols)}"
            )


def profile_dict(profile: CaptureProfile) -> dict:
    result = asdict(profile)
    result["config_path"] = str(profile.config_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--rover-id", required=True, type=int)
    parser.add_argument("--format", choices=("json", "null"), default="json")
    args = parser.parse_args()
    try:
        profile = resolve_profile(args.profile, args.rover_id)
    except ValueError as error:
        parser.error(str(error))
    values = profile_dict(profile)
    if args.format == "json":
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for key in (
            "config_path",
            "routine",
            "records_per_receiver",
            "expected_radios",
            "rx_transport",
            "data_version",
        ):
            print(values[key], end="\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

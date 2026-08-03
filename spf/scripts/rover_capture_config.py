"""Resolve and validate the canonical Rover 3.1 production capture config."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from spf.capture_schema import normalize_capture_config, validate_transport_schema


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1/capture_configs"
CANONICAL_CONFIGS = {
    1: "rover1_production_v7.yaml",
    2: "rover2_production_v7.yaml",
    3: "rover3_production_v7.yaml",
    4: "rover4_production_v7.yaml",
}
FIRMWARE_KEYS = (
    "release-tag",
    "device-fw",
    "asset-name",
    "image-url",
    "image-sha256",
    "firmware-git-sha",
    "gadget-git-sha",
    "boot-mode",
)
# Maps each pluto-firmware YAML key to its RoverCapturePlan attribute. Kept
# beside FIRMWARE_KEYS so the two cannot drift: firmware_block() below asserts
# that every key is mapped, so adding a field to FIRMWARE_KEYS without mapping
# it fails loudly instead of being silently dropped from firmware comparisons.
FIRMWARE_KEY_TO_PLAN_ATTR = {
    "release-tag": "firmware_release_tag",
    "device-fw": "firmware_device_fw",
    "asset-name": "firmware_asset_name",
    "image-url": "firmware_image_url",
    "image-sha256": "firmware_image_sha256",
    "firmware-git-sha": "firmware_git_sha",
    "gadget-git-sha": "gadget_git_sha",
    "boot-mode": "firmware_boot_mode",
}


@dataclass(frozen=True)
class RoverCapturePlan:
    rover_id: int
    config_path: str
    config_sha256: str
    routine: str
    records_per_receiver: int
    expected_radios: int
    rx_transport: str
    data_version: int
    firmware_release_tag: str
    firmware_asset_name: str
    firmware_image_url: str
    firmware_image_sha256: str
    firmware_git_sha: str
    gadget_git_sha: str
    firmware_boot_mode: str
    firmware_device_fw: str


def firmware_block(plan: RoverCapturePlan) -> dict[str, str]:
    """Rebuild a plan's ``pluto-firmware`` mapping, exactly as written in YAML.

    Derived from FIRMWARE_KEYS rather than hand-listed so that a field added to
    the contract is always included in firmware equality checks. Callers compare
    this against a config's own ``pluto-firmware`` block.
    """
    missing = [key for key in FIRMWARE_KEYS if key not in FIRMWARE_KEY_TO_PLAN_ATTR]
    if missing:
        raise RuntimeError(
            f"FIRMWARE_KEY_TO_PLAN_ATTR is missing entries for: {missing}"
        )
    return {key: getattr(plan, FIRMWARE_KEY_TO_PLAN_ATTR[key]) for key in FIRMWARE_KEYS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_path(rover_id: int) -> Path:
    try:
        filename = CANONICAL_CONFIGS[rover_id]
    except KeyError as exc:
        raise ValueError(f"unsupported rover_id: {rover_id}") from exc
    return CONFIG_ROOT / filename


def load_rover_capture_config(
    rover_id: int, config_override: str | Path | None = None
) -> tuple[Path, dict]:
    path = (
        Path(config_override).expanduser()
        if config_override is not None
        else canonical_config_path(rover_id)
    )
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"capture config does not exist: {path}")
    with path.open() as source:
        raw_config = yaml.safe_load(source)
    if not isinstance(raw_config, dict):
        raise ValueError(f"capture config must contain a YAML mapping: {path}")
    config = normalize_capture_config(raw_config)
    validate_production_config(config, path)
    return path, config


def validate_production_config(config: dict, path: Path | None = None) -> None:
    label = str(path) if path is not None else "capture config"
    validate_transport_schema(config)

    if config.get("data-version") != 7:
        raise ValueError(f"{label}: production Rover capture requires data-version: 7")

    receivers = config["receivers"]
    receiver_ports = [receiver.get("receiver-port") for receiver in receivers]
    if any(not isinstance(port, int) or port <= 0 for port in receiver_ports):
        raise ValueError(f"{label}: every receiver requires a positive receiver-port")
    if len(receiver_ports) != len(set(receiver_ports)):
        raise ValueError(f"{label}: receiver-port values must be unique")

    routine = config.get("routine")
    if not isinstance(routine, str) or not routine:
        raise ValueError(f"{label}: routine must be a non-empty string")
    records = config.get("n-records-per-receiver")
    if not isinstance(records, int) or records <= 0:
        raise ValueError(f"{label}: n-records-per-receiver must be a positive integer")

    firmware = config.get("pluto-firmware")
    if not isinstance(firmware, dict):
        raise ValueError(f"{label}: missing top-level pluto-firmware mapping")
    missing = [key for key in FIRMWARE_KEYS if not firmware.get(key)]
    if missing:
        raise ValueError(
            f"{label}: pluto-firmware is missing required fields: {missing}"
        )
    if firmware["boot-mode"] not in {"ram", "qspi"}:
        raise ValueError(
            f"{label}: pluto-firmware.boot-mode must be 'ram' or 'qspi'"
        )
    for key in ("image-sha256", "firmware-git-sha", "gadget-git-sha"):
        value = firmware[key]
        expected_length = 64 if key == "image-sha256" else 40
        if (
            not isinstance(value, str)
            or len(value) != expected_length
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ValueError(
                f"{label}: pluto-firmware.{key} must be a "
                f"{expected_length}-character hexadecimal digest"
            )


def resolve_capture_plan(
    rover_id: int, config_override: str | Path | None = None
) -> RoverCapturePlan:
    path, config = load_rover_capture_config(rover_id, config_override)
    receivers = config["receivers"]
    transports = {receiver["rx-transport"] for receiver in receivers}
    if len(transports) != 1:
        raise ValueError(f"{path}: receiver transports are not homogeneous")
    firmware = config["pluto-firmware"]
    return RoverCapturePlan(
        rover_id=rover_id,
        config_path=str(path),
        config_sha256=_sha256(path),
        routine=config["routine"],
        records_per_receiver=config["n-records-per-receiver"],
        expected_radios=len(receivers),
        rx_transport=transports.pop(),
        data_version=config["data-version"],
        firmware_release_tag=firmware["release-tag"],
        firmware_asset_name=firmware["asset-name"],
        firmware_image_url=firmware["image-url"],
        firmware_image_sha256=firmware["image-sha256"],
        firmware_git_sha=firmware["firmware-git-sha"],
        gadget_git_sha=firmware["gadget-git-sha"],
        firmware_boot_mode=firmware["boot-mode"],
        firmware_device_fw=firmware["device-fw"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rover-id", required=True, type=int)
    parser.add_argument("--config")
    parser.add_argument("--format", choices=("json", "null"), default="json")
    args = parser.parse_args()
    try:
        plan = resolve_capture_plan(args.rover_id, args.config)
    except ValueError as error:
        parser.error(str(error))
    values = asdict(plan)
    if args.format == "json":
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for key in values:
            print(values[key], end="\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate current rover pins and immutable historical calibration provenance.

Calibration configurations record the firmware that produced each historical
run. They must remain self-describing; promoting rover firmware must not rewrite
that evidence. Only the current rover configurations are required to agree with
one another.
"""

from pathlib import Path

import pytest
import yaml

from spf.scripts.rover_capture_config import (
    FIRMWARE_KEY_TO_PLAN_ATTR,
    FIRMWARE_KEYS,
    canonical_config_path,
    firmware_block,
    resolve_capture_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_CONFIG_DIR = REPO_ROOT / "spf/calibrations/dual_rx_gain_frequency/configs"
REFERENCE_ROVER_ID = 1


def _calibration_configs_with_firmware() -> list[Path]:
    paths = []
    for path in sorted(CALIBRATION_CONFIG_DIR.glob("*.yaml")):
        document = yaml.safe_load(path.read_text())
        if isinstance(document, dict) and document.get("pluto-firmware") is not None:
            paths.append(path)
    return paths


def test_calibration_config_dir_has_pinned_configs():
    """Guard the guard: if the glob stops matching, the tests below are vacuous."""
    assert _calibration_configs_with_firmware(), (
        f"no calibration config under {CALIBRATION_CONFIG_DIR} declares "
        "pluto-firmware; this test can no longer detect pin drift"
    )


@pytest.mark.parametrize(
    "config_path", _calibration_configs_with_firmware(), ids=lambda p: p.name
)
def test_historical_calibration_firmware_pin_is_self_describing(config_path: Path):
    """Historical runs retain a complete, internally consistent source pin."""
    actual = yaml.safe_load(config_path.read_text())["pluto-firmware"]
    assert set(actual) == set(FIRMWARE_KEYS)
    assert all(actual.values())
    assert actual["boot-mode"] in {"ram", "qspi"}
    assert len(actual["image-sha256"]) == 64
    assert len(actual["firmware-git-sha"]) == 40
    assert len(actual["gadget-git-sha"]) == 40
    assert actual["image-url"].endswith(
        f"/{actual['release-tag']}/{actual['asset-name']}"
    )


@pytest.mark.parametrize(
    "config_path", _calibration_configs_with_firmware(), ids=lambda p: p.name
)
def test_historical_direct_usb_protocol_is_complete_when_present(
    config_path: Path,
):
    """Older runs may predate direct USB; later runs record its full contract."""
    actual = yaml.safe_load(config_path.read_text()).get("direct-usb")
    if actual is None:
        return
    assert set(actual) == {
        "protocol-version",
        "gain-observation-interval-samples",
        "gain-observation-capacity",
    }
    assert all(isinstance(value, int) and value > 0 for value in actual.values())


def test_all_rover_configs_pin_identical_firmware():
    """The three rover configs must agree with each other, or 'the' pin is ambiguous."""
    blocks = {
        rover_id: yaml.safe_load(canonical_config_path(rover_id).read_text())[
            "pluto-firmware"
        ]
        for rover_id in (1, 2, 3, 4)
    }
    reference = blocks[REFERENCE_ROVER_ID]
    for rover_id, block in blocks.items():
        differing = {
            key: {rover_id: block.get(key), REFERENCE_ROVER_ID: reference.get(key)}
            for key in set(reference) | set(block)
            if block.get(key) != reference.get(key)
        }
        assert not differing, (
            f"rover {rover_id} pins different firmware than rover "
            f"{REFERENCE_ROVER_ID}: {differing}"
        )


def test_firmware_block_covers_every_contract_key():
    """firmware_block() must emit all of FIRMWARE_KEYS.

    The regression this prevents: `device-fw` was added to FIRMWARE_KEYS and to
    RoverCapturePlan, but the hand-written dict that rebuilt a plan's firmware
    mapping was not updated, so firmware equality checks silently stopped
    comparing it.
    """
    assert set(FIRMWARE_KEY_TO_PLAN_ATTR) == set(FIRMWARE_KEYS), (
        "FIRMWARE_KEY_TO_PLAN_ATTR and FIRMWARE_KEYS disagree; every contract "
        "key needs a RoverCapturePlan attribute"
    )

    block = firmware_block(resolve_capture_plan(REFERENCE_ROVER_ID))
    assert set(block) == set(FIRMWARE_KEYS)
    assert all(value for value in block.values()), f"empty firmware field in {block}"


def test_firmware_block_roundtrips_the_rover_config():
    """A plan rebuilt into a firmware mapping equals the YAML it was loaded from."""
    for rover_id in (1, 2, 3, 4):
        on_disk = yaml.safe_load(canonical_config_path(rover_id).read_text())[
            "pluto-firmware"
        ]
        rebuilt = firmware_block(resolve_capture_plan(rover_id))
        assert rebuilt == on_disk, (
            f"rover {rover_id}: firmware_block() does not round-trip its config; "
            f"rebuilt={rebuilt} on_disk={on_disk}"
        )

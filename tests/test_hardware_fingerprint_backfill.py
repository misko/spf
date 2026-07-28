import json

import numpy as np
import pytest

from spf.hardware_fingerprint import stable_identity_sha256
from spf.calibrations.dual_rx_gain_frequency.backfill_hardware_fingerprint import (
    BackfillError,
    inspect_store,
    run_backfill,
)
from spf.dataset.v7_data import v7rx_new_dataset
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


SERIAL = "104000f6ad020002fdff3a00bba2f096a1"


def _make_store(path):
    zarr = v7rx_new_dataset(
        filename=str(path),
        timesteps=2,
        buffer_size=4,
        n_receivers=1,
        config={"data-version": 7, "receivers": [{}]},
        chunk_size=1,
        compressor=None,
    )
    zarr.attrs.update(
        {
            "calibration_schema": "spf.calibration.dual_rx_gain_frequency",
            "calibration_schema_version": 1,
        }
    )
    receiver = zarr["receivers/r0"]
    receiver.attrs.update(
        {
            "sdr_serial": SERIAL,
            "firmware_git_sha": "old-acquisition-firmware",
        }
    )
    receiver.signal_matrix[:] = np.arange(2 * 2 * 4, dtype=np.float64).reshape(2, 2, 4)
    receiver.create_dataset(
        "sweep_completed",
        data=np.asarray([True, False]),
        chunks=(2,),
        compressor=None,
    )
    zarr.store.close()


def _fingerprint():
    stable_identity = {
        "pluto_serial": SERIAL,
        "spi_nor_unique_id_hmac_sha256": "b" * 64,
    }
    return {
        "schema": "spf.hardware_compatibility_fingerprint",
        "schema_version": 1,
        "fingerprint_timing": "post_firmware_before_recording",
        "acquisition_binding": True,
        "passive_observation": True,
        "tx_operations_performed": False,
        "2r2t_configured": True,
        "2r2t_functionally_verified": False,
        "host_boot_id": "BOOT-A",
        "fingerprint_session_id": "SESSION-A",
        "hmac_key_id": "a" * 16,
        "stable_identity": stable_identity,
        "stable_fingerprint_sha256": stable_identity_sha256(stable_identity),
        "attachment": {"usb_port_path": "1.1"},
        "firmware_session": {"firmware_git_sha": "current-firmware"},
        "direct_usb": {"protocol_min": 1, "protocol_max": 2},
        "device_facts": {"uboot_mode": "2r2t"},
        "compatibility": {"status": "compatible", "failures": []},
    }


def _ready(path):
    path.write_text(
        json.dumps(
            {
                "ready_manifest_version": 2,
                "fingerprint_session_id": "SESSION-A",
                "radios": [
                    {
                        "serial": SERIAL,
                        "hardware_fingerprint": _fingerprint(),
                    }
                ],
            }
        )
    )


def test_backfill_dry_run_then_apply_preserves_all_arrays_and_provenance(tmp_path):
    store = tmp_path / "calibration.v7.zarr"
    _make_store(store)
    ready = tmp_path / "ready.json"
    _ready(ready)
    before = inspect_store(store)

    dry_report = run_backfill(
        [tmp_path],
        ready_manifest=ready,
        apply=False,
        report_path=tmp_path / "dry-run.json",
        observed_at_unix_ns=100,
    )

    assert dry_report["summary"]["would_backfill"] == 1
    unchanged = inspect_store(store)
    assert "hardware_fingerprint_v1" not in unchanged["receiver_attrs"]

    report = run_backfill(
        [tmp_path],
        ready_manifest=ready,
        apply=True,
        report_path=tmp_path / "apply.json",
        observed_at_unix_ns=100,
    )

    assert report["summary"]["backfilled"] == 1
    after = inspect_store(store)
    assert before["stored_array_sha256"] == after["stored_array_sha256"]
    assert before["signal_matrix_shape"] == after["signal_matrix_shape"]
    assert before["completed_frames"] == after["completed_frames"]
    assert after["receiver_attrs"]["firmware_git_sha"] == ("old-acquisition-firmware")
    fingerprint = after["receiver_attrs"]["hardware_fingerprint_v1"]
    assert fingerprint["fingerprint_timing"] == "post_run_backfill"
    assert fingerprint["acquisition_binding"] is False
    assert fingerprint["matched_by"] == "pluto_serial"
    assert "firmware_session" not in fingerprint
    assert (
        fingerprint["post_run_observation"]["firmware_session"]["firmware_git_sha"]
        == "current-firmware"
    )


def test_backfill_is_idempotent_across_observation_times(tmp_path):
    store = tmp_path / "calibration.v7.zarr"
    _make_store(store)
    ready = tmp_path / "ready.json"
    _ready(ready)
    run_backfill(
        [store],
        ready_manifest=ready,
        apply=True,
        report_path=tmp_path / "first.json",
        observed_at_unix_ns=100,
    )

    second = run_backfill(
        [store],
        ready_manifest=ready,
        apply=True,
        report_path=tmp_path / "second.json",
        observed_at_unix_ns=200,
    )

    assert second["summary"]["already_current"] == 1


def test_backfill_rejects_serial_without_current_fingerprint(tmp_path):
    store = tmp_path / "calibration.v7.zarr"
    _make_store(store)
    ready = tmp_path / "ready.json"
    ready.write_text(
        json.dumps(
            {
                "ready_manifest_version": 2,
                "fingerprint_session_id": "SESSION-A",
                "radios": [],
            }
        )
    )

    with pytest.raises(BackfillError, match="failed preflight"):
        run_backfill(
            [store],
            ready_manifest=ready,
            apply=True,
            report_path=tmp_path / "failed.json",
        )

    zarr = zarr_open_from_lmdb_store(str(store))
    try:
        assert "hardware_fingerprint_v1" not in zarr["receivers/r0"].attrs
    finally:
        zarr.store.close()


def test_backfill_preflights_entire_batch_before_first_mutation(tmp_path):
    eligible = tmp_path / "eligible" / "calibration.v7.zarr"
    ineligible = tmp_path / "ineligible" / "calibration.v7.zarr"
    eligible.parent.mkdir()
    ineligible.parent.mkdir()
    _make_store(eligible)
    _make_store(ineligible)
    zarr = zarr_open_from_lmdb_store(str(ineligible), mode="rw")
    try:
        zarr["receivers/r0"].attrs["sdr_serial"] = "UNKNOWN-SERIAL"
    finally:
        zarr.store.close()
    ready = tmp_path / "ready.json"
    _ready(ready)
    report_path = tmp_path / "failed-preflight.json"

    with pytest.raises(BackfillError, match="no stores were modified"):
        run_backfill(
            [tmp_path],
            ready_manifest=ready,
            apply=True,
            report_path=report_path,
        )

    assert "hardware_fingerprint_v1" not in inspect_store(eligible)["receiver_attrs"]
    report = json.loads(report_path.read_text())
    assert report["phase"] == "preflight_failed"
    assert report["summary"] == {"failed": 1, "preflight_passed": 1}

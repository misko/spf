"""Safely add post-run hardware fingerprints to calibration V7 stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np

from spf.hardware_fingerprint import public_fingerprint_copy
from spf.scripts.pluto_ready_manifest import fingerprint_for_serial, load_manifest
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


CALIBRATION_SCHEMA = "spf.calibration.dual_rx_gain_frequency"
BACKFILL_REPORT_SCHEMA = "spf.calibration.hardware_fingerprint_backfill"
BACKFILL_REPORT_VERSION = 1


class BackfillError(RuntimeError):
    pass


def discover_calibration_stores(roots: Iterable[Path]) -> list[Path]:
    stores: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.name.endswith(".zarr") and (root / "data.mdb").is_file():
            stores.add(root.resolve())
            continue
        if root.is_dir():
            stores.update(
                path.resolve()
                for path in root.rglob("*.zarr")
                if (path / "data.mdb").is_file()
            )
    return sorted(stores)


def stored_array_sha256(store) -> str:
    """Hash stored Zarr schemas/chunks while excluding mutable attributes.

    Calibration stores can preallocate hundreds of gigabytes of logical zero
    frames while physically containing only completed chunks. Hashing raw
    store entries proves that the encoded array schemas and every materialized
    chunk remain byte-identical without synthesizing those unwritten zeros.
    """

    digest = hashlib.sha256()
    for key in sorted(store.keys()):
        if key == ".zattrs" or key.endswith("/.zattrs"):
            continue
        encoded_key = key.encode("utf-8")
        value = store[key]
        digest.update(len(encoded_key).to_bytes(8, "little"))
        digest.update(encoded_key)
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    return digest.hexdigest()


def historical_fingerprint(
    source: dict[str, Any],
    *,
    observed_at_unix_ns: int,
) -> dict[str, Any]:
    source = public_fingerprint_copy(source)
    observation = {
        key: source[key]
        for key in (
            "host_boot_id",
            "fingerprint_session_id",
            "attachment",
            "firmware_session",
            "direct_usb",
            "device_facts",
        )
        if key in source
    }
    return public_fingerprint_copy(
        {
            "schema": source["schema"],
            "schema_version": source["schema_version"],
            "fingerprint_timing": "post_run_backfill",
            "acquisition_binding": False,
            "matched_by": "pluto_serial",
            "passive_observation": source.get("passive_observation", True),
            "tx_operations_performed": False,
            "2r2t_configured": source.get("2r2t_configured"),
            "2r2t_functionally_verified": False,
            "backfill_observed_at_unix_ns": observed_at_unix_ns,
            "hmac_key_id": source["hmac_key_id"],
            "stable_identity": source["stable_identity"],
            "stable_fingerprint_sha256": source["stable_fingerprint_sha256"],
            "compatibility": source["compatibility"],
            "post_run_observation": observation,
        }
    )


def _atomic_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w") as destination:
            json.dump(document, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def inspect_store(path: Path) -> dict[str, Any]:
    zarr = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        if zarr.attrs.get("calibration_schema") != CALIBRATION_SCHEMA:
            raise BackfillError(f"{path}: not a dual-RX gain/frequency calibration")
        receiver_names = sorted(zarr["receivers"].group_keys())
        if receiver_names != ["r0"]:
            raise BackfillError(
                f"{path}: expected one calibration receiver, found {receiver_names}"
            )
        receiver = zarr["receivers/r0"]
        serial = receiver.attrs.get("sdr_serial")
        if not isinstance(serial, str) or not serial:
            raise BackfillError(f"{path}: receiver serial is missing")
        return {
            "path": str(path),
            "serial": serial,
            "root_attrs": dict(zarr.attrs),
            "receiver_attrs": dict(receiver.attrs),
            "stored_array_sha256": stored_array_sha256(zarr.store),
            "signal_matrix_shape": list(receiver["signal_matrix"].shape),
            "completed_frames": (
                int(np.count_nonzero(receiver["sweep_completed"][:]))
                if "sweep_completed" in receiver
                else None
            ),
        }
    finally:
        zarr.store.close()


def backfill_store(
    inventory: dict[str, Any],
    *,
    fingerprint: dict[str, Any],
    apply: bool,
) -> dict[str, Any]:
    path = Path(inventory["path"])
    existing = inventory["receiver_attrs"].get("hardware_fingerprint_v1")
    if existing is not None:
        if (
            isinstance(existing, dict)
            and existing.get("fingerprint_timing") == "post_run_backfill"
            and existing.get("stable_fingerprint_sha256")
            == fingerprint.get("stable_fingerprint_sha256")
        ):
            return {
                "path": str(path),
                "serial": inventory["serial"],
                "status": "already_current",
                "stored_array_sha256_before": inventory["stored_array_sha256"],
                "stored_array_sha256_after": inventory["stored_array_sha256"],
            }
        raise BackfillError(f"{path}: conflicting hardware fingerprint already exists")
    if not apply:
        return {
            "path": str(path),
            "serial": inventory["serial"],
            "status": "would_backfill",
            "stored_array_sha256_before": inventory["stored_array_sha256"],
        }

    zarr = zarr_open_from_lmdb_store(str(path), mode="rw")
    try:
        receiver = zarr["receivers/r0"]
        if receiver.attrs.get("sdr_serial") != inventory["serial"]:
            raise BackfillError(f"{path}: serial changed after inventory")
        receiver.attrs.update(
            {
                "hardware_fingerprint_schema_version": fingerprint["schema_version"],
                "hardware_fingerprint_v1": fingerprint,
            }
        )
    finally:
        zarr.store.close()

    verified = inspect_store(path)
    if verified["stored_array_sha256"] != inventory["stored_array_sha256"]:
        raise BackfillError(f"{path}: stored array content changed during backfill")
    if (
        verified["signal_matrix_shape"] != inventory["signal_matrix_shape"]
        or verified["completed_frames"] != inventory["completed_frames"]
    ):
        raise BackfillError(f"{path}: array shape or completed count changed")
    if verified["receiver_attrs"].get("hardware_fingerprint_v1") != fingerprint:
        raise BackfillError(f"{path}: fingerprint did not round-trip")
    return {
        "path": str(path),
        "serial": inventory["serial"],
        "status": "backfilled",
        "stored_array_sha256_before": inventory["stored_array_sha256"],
        "stored_array_sha256_after": verified["stored_array_sha256"],
    }


def run_backfill(
    roots: Iterable[Path],
    *,
    ready_manifest: Path,
    apply: bool,
    report_path: Path,
    observed_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(ready_manifest)
    if manifest.get("ready_manifest_version") != 2:
        raise BackfillError("backfill requires a session-bound ready manifest v2")
    observed_ns = time.time_ns() if observed_at_unix_ns is None else observed_at_unix_ns
    stores = discover_calibration_stores(roots)
    if not stores:
        raise BackfillError("no calibration Zarr stores were found")
    report: dict[str, Any] = {
        "schema": BACKFILL_REPORT_SCHEMA,
        "schema_version": BACKFILL_REPORT_VERSION,
        "mode": "apply" if apply else "dry_run",
        "observed_at_unix_ns": observed_ns,
        "ready_manifest": str(ready_manifest),
        "ready_fingerprint_session_id": manifest.get("fingerprint_session_id"),
        "phase": "preflight",
        "stores": [],
    }
    prepared: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in stores:
        try:
            inventory = inspect_store(path)
            source = fingerprint_for_serial(manifest, inventory["serial"])
            if source is None:
                raise BackfillError(
                    f"{path}: no unique current fingerprint for "
                    f"{inventory['serial']}"
                )
            fingerprint = historical_fingerprint(
                source,
                observed_at_unix_ns=observed_ns,
            )
            prepared.append((inventory, fingerprint))
            report["stores"].append(
                {
                    "path": str(path),
                    "serial": inventory["serial"],
                    "status": "preflight_passed",
                    "stored_array_sha256_before": inventory["stored_array_sha256"],
                    "original_root_attrs": inventory["root_attrs"],
                    "original_receiver_attrs": inventory["receiver_attrs"],
                }
            )
        except (BackfillError, OSError, ValueError) as error:
            report["stores"].append(
                {
                    "path": str(path),
                    "status": "failed",
                    "error": str(error),
                }
            )

    failed = sum(row["status"] == "failed" for row in report["stores"])
    if failed:
        report["phase"] = "preflight_failed"
        report["summary"] = {
            "preflight_passed": len(prepared),
            "failed": failed,
        }
        _atomic_json(report_path, report)
        raise BackfillError(
            f"{failed} stores failed preflight; no stores were modified; "
            f"see {report_path}"
        )

    if not apply:
        for row in report["stores"]:
            row["status"] = "would_backfill"
            row.pop("original_root_attrs", None)
            row.pop("original_receiver_attrs", None)
        report["phase"] = "dry_run_complete"
        report["summary"] = {
            "would_backfill": len(prepared),
            "failed": 0,
        }
        _atomic_json(report_path, report)
        return report

    # Persist every store's original attributes and logical-content digest before
    # the first mutation. This is the recovery record if the process or host
    # stops between independent LMDB store commits.
    report["phase"] = "apply_in_progress"
    _atomic_json(report_path, report)
    results_by_path = {row["path"]: row for row in report["stores"]}
    for inventory, fingerprint in prepared:
        path = inventory["path"]
        recovery = results_by_path[path]
        try:
            result = backfill_store(
                inventory,
                fingerprint=fingerprint,
                apply=True,
            )
            recovery.update(result)
        except (BackfillError, OSError, ValueError) as error:
            recovery.update(
                {
                    "status": "failed",
                    "error": str(error),
                }
            )
        _atomic_json(report_path, report)

    report["summary"] = {
        status: sum(row["status"] == status for row in report["stores"])
        for status in ("backfilled", "already_current", "failed")
    }
    report["phase"] = (
        "apply_failed" if report["summary"]["failed"] else "apply_complete"
    )
    _atomic_json(report_path, report)
    if report["summary"]["failed"]:
        raise BackfillError(
            f"{report['summary']['failed']} stores failed during apply; "
            f"see recovery data in {report_path}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument(
        "--ready-manifest",
        type=Path,
        default=Path("/run/spf/direct_usb_ready.json"),
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Mutate eligible stores; the default is a dry run.",
    )
    args = parser.parse_args()
    try:
        report = run_backfill(
            args.roots,
            ready_manifest=args.ready_manifest,
            apply=args.apply,
            report_path=args.report,
        )
    except BackfillError as error:
        parser.error(str(error))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

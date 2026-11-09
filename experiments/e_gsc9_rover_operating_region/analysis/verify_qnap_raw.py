#!/usr/bin/env python3
"""Verify every current local E-GSC9 raw artifact against its QNAP copy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import lmdb


SCHEMA = "spf.experiment.e_gsc9.qnap_raw_verification"
SCHEMA_VERSION = 1
IGNORED_LMDB_FILES = {"data.mdb", "lock.mdb"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lmdb(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    environment = lmdb.open(
        str(path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=True,
        max_readers=32,
    )
    try:
        with environment.begin(buffers=True) as transaction:
            entries = 0
            payload_bytes = 0
            for key, value in transaction.cursor():
                key_bytes = bytes(key)
                value_bytes = bytes(value)
                digest.update(len(key_bytes).to_bytes(8, "little"))
                digest.update(key_bytes)
                digest.update(len(value_bytes).to_bytes(8, "little"))
                digest.update(value_bytes)
                payload_bytes += len(key_bytes) + len(value_bytes)
                entries += 1
            stat_entries = int(transaction.stat()["entries"])
    finally:
        environment.close()
    if entries != stat_entries:
        raise ValueError(
            f"{path}: iterated {entries} entries, LMDB reports {stat_entries}"
        )
    return {
        "logical_sha256": digest.hexdigest(),
        "entries": entries,
        "logical_payload_bytes": payload_bytes,
    }


def _write_atomic(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(document, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _capture_roots(base: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(base.glob("e_gsc9_*"))
        if path.is_dir() and any(path.glob("*/calibration.v7.zarr/data.mdb"))
    }


def _store_pairs(
    local_roots: dict[str, Path], qnap_roots: dict[str, Path]
) -> list[tuple[str, Path, Path]]:
    pairs = []
    for root_name, local_root in local_roots.items():
        for local_store in sorted(local_root.glob("*/calibration.v7.zarr")):
            relative = local_store.relative_to(local_root)
            pairs.append(
                (
                    f"{root_name}/{relative}",
                    local_store,
                    qnap_roots[root_name] / relative,
                )
            )
    return pairs


def _verify_store(item: tuple[str, Path, Path]) -> dict[str, Any]:
    relative, local, qnap = item
    if not qnap.is_dir():
        raise FileNotFoundError(f"missing QNAP LMDB store: {qnap}")
    local_hash = _sha256_lmdb(local)
    qnap_hash = _sha256_lmdb(qnap)
    matches = local_hash == qnap_hash
    print(
        f"{'PASS' if matches else 'FAIL'} LMDB {relative}: "
        f"{local_hash['entries']} entries, {local_hash['logical_payload_bytes']} bytes",
        flush=True,
    )
    return {
        "path": relative,
        "local": local_hash,
        "qnap": qnap_hash,
        "match": matches,
    }


def _sidecar_paths(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_LMDB_FILES
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--qnap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    local_roots = _capture_roots(args.local_root)
    qnap_roots = _capture_roots(args.qnap_root)
    missing_roots = sorted(set(local_roots) - set(qnap_roots))
    if missing_roots:
        raise FileNotFoundError(f"QNAP is missing capture roots: {missing_roots}")
    qnap_roots = {name: qnap_roots[name] for name in local_roots}

    stores = []
    pairs = _store_pairs(local_roots, qnap_roots)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_verify_store, pair) for pair in pairs]
        for future in as_completed(futures):
            stores.append(future.result())
    stores.sort(key=lambda row: row["path"])

    sidecars = []
    qnap_only_sidecars = []
    for root_name, local_root in local_roots.items():
        qnap_root = qnap_roots[root_name]
        local_paths = _sidecar_paths(local_root)
        qnap_paths = _sidecar_paths(qnap_root)
        missing = sorted(str(path) for path in local_paths - qnap_paths)
        if missing:
            raise FileNotFoundError(f"{root_name}: QNAP is missing sidecars: {missing}")
        for relative in sorted(local_paths):
            local_hash = _sha256_file(local_root / relative)
            qnap_hash = _sha256_file(qnap_root / relative)
            sidecars.append(
                {
                    "path": f"{root_name}/{relative}",
                    "sha256": local_hash,
                    "match": local_hash == qnap_hash,
                }
            )
        qnap_only_sidecars.extend(
            f"{root_name}/{relative}" for relative in sorted(qnap_paths - local_paths)
        )

    all_match = all(row["match"] for row in stores + sidecars)
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix_ns": time.time_ns(),
        "local_root": str(args.local_root.resolve()),
        "qnap_root": str(args.qnap_root.resolve()),
        "capture_roots": sorted(local_roots),
        "capture_root_count": len(local_roots),
        "lmdb_store_count": len(stores),
        "sidecar_count": len(sidecars),
        "qnap_only_sidecars": sorted(qnap_only_sidecars),
        "lmdb_stores": stores,
        "sidecars": sidecars,
        "all_local_raw_present_and_identical_on_qnap": all_match,
    }
    _write_atomic(args.output, result)
    print(
        f"{'PASS' if all_match else 'FAIL'}: {len(local_roots)} roots, "
        f"{len(stores)} LMDB stores, {len(sidecars)} sidecars",
        flush=True,
    )
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Where a filter sweep's results go.

The runner previously wrote every result straight into a DynamoDB table, with
the local file path behind a hard-coded ``use_file_system = False``. That made a
local sweep impossible without AWS credentials, and left the runner and
``run_filters_report.py`` -- which reads ``.pkl`` files out of the work dir --
wired to two different sinks, so the local reporter had nothing to read.

Both sinks now sit behind one interface, and **local is the default**. The
DynamoDB path is preserved for the AWS Batch driver but is opt-in and imports
boto3 lazily, so no local run touches the network.
"""

import logging
import os
import pickle
from decimal import Decimal


def float_to_decimal(obj):
    """Recursively convert floats to Decimal, which is what DynamoDB accepts."""
    if isinstance(obj, float):
        # via str: Decimal(0.1) carries the binary float's full error
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(v) for v in obj]
    return obj


class ResultsStore:
    """Sink for one sweep. ``key`` is the workdir-relative result path."""

    def already_processed(self):
        """Keys already present, so a resumed sweep can skip them."""
        raise NotImplementedError

    def put(self, key, results):
        raise NotImplementedError


class LocalResultsStore(ResultsStore):
    """One pickle per job under ``workdir``, laid out as the reporter expects."""

    def __init__(self, workdir):
        self.workdir = workdir

    def path_for(self, key):
        return os.path.join(self.workdir, key)

    def already_processed(self):
        found = set()
        for dirpath, _, filenames in os.walk(self.workdir):
            for name in filenames:
                if name.endswith(".pkl"):
                    found.add(
                        os.path.relpath(os.path.join(dirpath, name), self.workdir)
                    )
        return found

    def put(self, key, results):
        path = self.path_for(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # write-then-rename: a sweep killed mid-write must not leave a truncated
        # pickle that the reporter would later fail on
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(results, f)
        os.rename(tmp, path)


class DynamoDBResultsStore(ResultsStore):
    """The AWS Batch sink. boto3 is imported lazily so local runs never need it."""

    def __init__(self, workdir, table_name="filter_metrics"):
        import boto3

        self.workdir = workdir
        self.table = boto3.resource("dynamodb").Table(table_name)

    def already_processed(self):
        from spf.filters.run_filters_on_data_b2 import (
            get_all_items_by_bucket_scan_full_name,
        )

        return {
            item["full_name"]
            for item in get_all_items_by_bucket_scan_full_name(self.workdir)
        }

    def put(self, key, results):
        self.table.put_item(
            Item={
                "bucket": self.workdir,
                "full_name": key,
                "results": float_to_decimal(results),
            }
        )


def make_results_store(backend, workdir):
    if backend == "local":
        return LocalResultsStore(workdir)
    if backend == "dynamodb":
        logging.info("results backend: dynamodb (network writes enabled)")
        return DynamoDBResultsStore(workdir)
    raise ValueError(f"unknown results backend {backend!r}; expected local or dynamodb")

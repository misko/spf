import csv

from spf.scripts.validate_soak_resources import validate


def _write_rounds(path):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            delimiter="\t",
            fieldnames=(
                "round",
                "cases",
                "started_unix",
                "finished_unix",
                "status",
                "artifact_kib",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "round": 1,
                "cases": "sigterm:1",
                "started_unix": 100,
                "finished_unix": 200,
                "status": 0,
                "artifact_kib": 1,
            }
        )


def _write_resources(path, anon_values, available_kib=1_000_000):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "timestamp_unix",
                "rss_kib",
                "rss_anon_kib",
                "available_kib",
                "artifact_kib",
            ),
        )
        writer.writeheader()
        for index, anon in enumerate(anon_values):
            writer.writerow(
                {
                    "timestamp_unix": 110 + index,
                    "rss_kib": anon + 100,
                    "rss_anon_kib": anon,
                    "available_kib": available_kib,
                    "artifact_kib": 200 + index,
                }
            )


def test_soak_resources_accept_bounded_lifecycle_churn(tmp_path):
    resources = tmp_path / "resources.csv"
    rounds = tmp_path / "rounds.tsv"
    _write_resources(resources, [10_000, 500_000, 200_000, 20_000, 450_000])
    _write_rounds(rounds)

    result = validate(resources, rounds)

    assert result["status"] == "pass"
    assert result["completed_rounds"] == 1
    assert result["rounds"][0]["sample_count"] == 5


def test_soak_resources_reject_memory_pressure_and_no_recovery(tmp_path):
    resources = tmp_path / "resources.csv"
    rounds = tmp_path / "rounds.tsv"
    _write_resources(
        resources,
        [1_100_000] * 5,
        available_kib=100_000,
    )
    _write_rounds(rounds)

    result = validate(resources, rounds)

    assert result["status"] == "fail"
    assert len(result["failures"]) == 3

import csv

from spf.scripts.validate_soak_resources import validate


def _write_rounds(path, windows=((1, 100, 200),)):
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
        for round_number, started, finished in windows:
            writer.writerow(
                {
                    "round": round_number,
                    "cases": "sigterm:1",
                    "started_unix": started,
                    "finished_unix": finished,
                    "status": 0,
                    "artifact_kib": 1,
                }
            )


def _write_resources(path, anon_values, available_kib=1_000_000, timestamps=None):
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
        if timestamps is None:
            timestamps = [110 + index for index in range(len(anon_values))]
        for index, (timestamp, anon) in enumerate(zip(timestamps, anon_values, strict=True)):
            writer.writerow(
                {
                    "timestamp_unix": timestamp,
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
    assert result["rounds"][0]["minimum_anon_mib"] == 10_000 / 1024
    assert result["rounds"][0]["post_peak_minimum_anon_mib"] == 20_000 / 1024


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
    assert len(result["failures"]) == 4


def test_soak_resources_requires_recovery_in_every_completed_round(tmp_path):
    resources = tmp_path / "resources.csv"
    rounds = tmp_path / "rounds.tsv"
    _write_resources(
        resources,
        [10_000, 500_000, 200_000, 10_000, 500_000, 450_000],
        timestamps=[110, 120, 130, 205, 210, 220],
    )
    _write_rounds(rounds, windows=((1, 100, 200), (2, 201, 300)))

    result = validate(resources, rounds)

    assert result["status"] == "fail"
    assert result["failures"] == [
        "round 2 anonymous RSS never recovered below 384.0 MiB after its peak"
    ]
    assert result["rounds"][0]["minimum_anon_mib"] < 384
    assert result["rounds"][1]["minimum_anon_mib"] < 384
    assert result["rounds"][1]["post_peak_minimum_anon_mib"] > 384

import csv

from spf.scripts.validate_resource_samples import validate


FIELDS = (
    "timestamp_unix",
    "pid",
    "rss_kib",
    "rss_anon_kib",
    "rss_file_kib",
    "vmsize_kib",
    "available_kib",
    "artifact_kib",
)


def _write_samples(path, anonymous_mib, available_mib=1024):
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        for index, value in enumerate(anonymous_mib):
            writer.writerow(
                {
                    "timestamp_unix": 1000 + index * 30,
                    "pid": 123,
                    "rss_kib": (value + 500) * 1024,
                    "rss_anon_kib": value * 1024,
                    "rss_file_kib": 500 * 1024,
                    "vmsize_kib": 1000 * 1024,
                    "available_kib": available_mib * 1024,
                    "artifact_kib": index * 4096,
                }
            )


def test_accepts_large_import_warmup_followed_by_plateau(tmp_path):
    path = tmp_path / "resources.csv"
    _write_samples(path, [10, 100, 300, 302, 301, 303, 302, 304, 303, 302])

    report = validate(path)

    assert report["status"] == "pass"
    assert report["post_warmup_anon_range_mib"] == 4


def test_rejects_unbounded_post_warmup_anonymous_growth(tmp_path):
    path = tmp_path / "resources.csv"
    _write_samples(path, [10, 100, 200, 250, 300, 350, 400, 450, 500, 550])

    report = validate(path, maximum_anon_range_mib=100)

    assert report["status"] == "fail"
    assert "anonymous RSS range" in report["failures"][0]


def test_rejects_low_available_host_memory(tmp_path):
    path = tmp_path / "resources.csv"
    _write_samples(path, [300] * 10, available_mib=64)

    report = validate(path)

    assert report["status"] == "fail"
    assert "available memory" in report["failures"][0]

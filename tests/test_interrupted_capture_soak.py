import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "data_collection/rover/rover_v3.1/run_interrupted_capture_soak.sh"
)


def test_interruption_soak_dry_run_rotates_bounded_matrices(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "SPF_INTERRUPT_SOAK_DRY_RUN": "1",
            "SPF_INTERRUPT_SOAK_SECONDS": "86400",
            "SPF_INTERRUPT_SOAK_MAX_ROUNDS": "4",
            "SPF_INTERRUPT_SOAK_MIN_FREE_GIB": "0",
            "SPF_INTERRUPT_SOAK_OUTPUT_ROOT": str(tmp_path),
        }
    )

    completed = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    run_root = next(tmp_path.iterdir())
    assert (run_root / "PASS").read_text() == "PASS\n"
    rows = (run_root / "rounds.tsv").read_text().splitlines()
    assert len(rows) == 5
    assert all("sigint:" in row and "sigterm:" in row for row in rows[1:])
    assert all("sigkill:" in row for row in rows[1:])
    assert len({row.split("\t")[1] for row in rows[1:]}) == 4


def test_interruption_soak_exposes_safety_controls():
    source = SCRIPT.read_text()

    assert "SPF_INTERRUPT_SOAK_SECONDS" in source
    assert "SPF_INTERRUPT_SOAK_MIN_FREE_GIB" in source
    assert "STOPPED_BY_FILE" in source
    assert "SPF_INTERRUPT_CLEAN_RECORDS" in source
    assert "if (( status != 0 ))" in source
    assert "source_diff_sha256" in source


def test_interruption_soak_fails_if_source_changes_between_rounds(tmp_path):
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(source_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(source_repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source_repo), "config", "user.name", "Test"],
        check=True,
    )
    tracked = source_repo / "tracked.txt"
    tracked.write_text("before\n")
    subprocess.run(["git", "-C", str(source_repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(source_repo), "commit", "-qm", "baseline"], check=True
    )

    campaign = tmp_path / "campaign.sh"
    campaign.write_text(
        f"""#!/usr/bin/env bash
set -eu
printf 'after\\n' >{str(tracked)!r}
"""
    )
    campaign.chmod(0o755)
    output = tmp_path / "output"
    environment = os.environ.copy()
    environment.update(
        {
            "SPF_INTERRUPT_REPO_ROOT": str(source_repo),
            "SPF_INTERRUPT_CAMPAIGN": str(campaign),
            "SPF_INTERRUPT_SOAK_SECONDS": "86400",
            "SPF_INTERRUPT_SOAK_MAX_ROUNDS": "2",
            "SPF_INTERRUPT_SOAK_MIN_FREE_GIB": "0",
            "SPF_INTERRUPT_SOAK_CLEAN_RECORDS": "1",
            "SPF_INTERRUPT_SOAK_OUTPUT_ROOT": str(output),
        }
    )

    completed = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    run_root = next(output.iterdir())
    failure = (run_root / "FAILED").read_text()
    assert failure.startswith("SOURCE_CHANGED\n")
    assert "expected_diff_sha256=" in failure
    assert "observed_diff_sha256=" in failure
    assert len((run_root / "rounds.tsv").read_text().splitlines()) == 1

"""Guards for stage_captures.sh — copying captures off a rover.

A recorded session is a FAMILY: `<capture>.zarr` + `<capture>.yaml` + `.log`
(all three carrying `.tmp` until the run finalizes). On 2026-08-04 a staging copy
took only the `.zarr` directories and the TX/RX merge could not run, because it
reads the RX `.yaml` for the receiver/antenna config.

No rover and no ssh: the script takes a local source directory as well as a
`host:dir` one, so every grouping and verification rule is exercised against a
tmpdir. Remote is the same code path with `ls` and `rsync` reaching over ssh.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "data_collection/rover/rover_v3.1/stage_captures.sh"


def run(*args: str):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=60
    )


def make_capture(directory: Path, name: str, members=("zarr", "yaml", "log"),
                 tmp: str = "") -> None:
    """Write one capture family into `directory`."""
    for member in members:
        target = directory / f"{name}.{member}{tmp}"
        if member == "zarr":
            target.mkdir(parents=True)
            (target / "data.mdb").write_bytes(b"payload")
        else:
            target.write_text(f"{name} {member}\n")


@pytest.fixture
def source(tmp_path) -> Path:
    src = tmp_path / "rover_temp"
    src.mkdir()
    return src


@pytest.fixture
def dest(tmp_path) -> Path:
    return tmp_path / "staging"


# ----------------------------------------------------------- the whole family ---


def test_the_yaml_and_log_come_along_with_the_zarr(source, dest):
    """The bug: `cp *.zarr` stages a capture the merge cannot read."""
    make_capture(source, "rover_2026_08_04_23_21_14_tag_RO1")

    result = run("--from", str(source), "--to", str(dest))

    assert result.returncode == 0, result.stdout + result.stderr
    for member in ("zarr", "yaml", "log"):
        assert (dest / f"rover_2026_08_04_23_21_14_tag_RO1.{member}").exists(), member


def test_the_zarr_is_copied_recursively(source, dest):
    """A .zarr is a DIRECTORY (an LMDB store).

    rsync's --files-from cancels the -r that -a would imply, so without an
    explicit -r the store lands as an empty directory -- a corruption that only
    surfaces much later, when something tries to open it.
    """
    make_capture(source, "cap_tag_RO1")

    assert run("--from", str(source), "--to", str(dest)).returncode == 0
    assert (dest / "cap_tag_RO1.zarr/data.mdb").read_bytes() == b"payload"


def test_an_unfinalized_capture_keeps_its_tmp_family_together(source, dest):
    make_capture(source, "cap_tag_RO3", tmp=".tmp")

    result = run("--from", str(source), "--to", str(dest))

    assert result.returncode == 0, result.stdout + result.stderr
    for member in ("zarr", "yaml", "log"):
        assert (dest / f"cap_tag_RO3.{member}.tmp").exists(), member


def test_a_tmp_zarr_is_not_paired_with_a_finalized_yaml(source, dest):
    """Attaching the wrong config to a capture is worse than missing one."""
    make_capture(source, "cap_tag_RO1", members=("zarr",), tmp=".tmp")
    (source / "cap_tag_RO1.yaml").write_text("a different run\n")

    result = run("--from", str(source), "--to", str(dest))

    assert result.returncode != 0
    assert "NO .yaml" in result.stdout
    assert not (dest / "cap_tag_RO1.yaml.tmp").exists()


def test_unrelated_files_are_left_alone(source, dest):
    make_capture(source, "cap_tag_RO1")
    (source / "rover_receiver_config_pi_3mhz_43mm.yaml.bak").write_text("x\n")
    (source / "nohup.out").write_text("x\n")

    assert run("--from", str(source), "--to", str(dest)).returncode == 0
    assert not (dest / "nohup.out").exists()


# ------------------------------------------------------------------- report ---


def test_a_capture_with_no_sidecar_is_reported_and_fails_the_run(source, dest):
    make_capture(source, "good_tag_RO1")
    make_capture(source, "bare_tag_RO1", members=("zarr",))

    result = run("--from", str(source), "--to", str(dest))

    assert result.returncode != 0
    assert "bare_tag_RO1" in result.stdout
    assert "NO .yaml" in result.stdout
    # The good capture is still staged: a bad neighbour must not cost the rest.
    assert (dest / "good_tag_RO1.yaml").exists()


def test_a_missing_log_is_a_warning_not_a_failure(source, dest):
    """The merge never reads the .log; losing it costs the field record only."""
    make_capture(source, "cap_tag_RO1", members=("zarr", "yaml"))

    result = run("--from", str(source), "--to", str(dest))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no .log" in result.stdout


def test_every_incomplete_capture_is_listed_not_just_the_first(source, dest):
    for name in ("a_tag_RO1", "b_tag_RO1", "c_tag_RO3"):
        make_capture(source, name, members=("zarr",))

    result = run("--from", str(source), "--to", str(dest))

    assert result.returncode != 0
    for name in ("a_tag_RO1", "b_tag_RO1", "c_tag_RO3"):
        assert name in result.stdout, name


# -------------------------------------------------------------------- match ---


def test_match_selects_whole_families_by_capture_name(source, dest):
    make_capture(source, "cap_tag_RO1")
    make_capture(source, "cap_tag_RO3")

    result = run("--from", str(source), "--to", str(dest), "--match", "*tag_RO1*")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "cap_tag_RO1.yaml").exists(), "the glob must match the FAMILY"
    assert not (dest / "cap_tag_RO3.zarr").exists()


def test_a_match_that_selects_nothing_says_so(source, dest):
    make_capture(source, "cap_tag_RO1")

    result = run("--from", str(source), "--to", str(dest), "--match", "*RO9*")

    assert result.returncode == 0
    assert "nothing to stage" in result.stdout


# ------------------------------------------------------------------ dry run ---


def test_dry_run_copies_nothing(source, dest):
    make_capture(source, "cap_tag_RO1")

    result = run("--from", str(source), "--to", str(dest), "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cap_tag_RO1.yaml" in result.stdout
    assert not dest.exists(), "--dry-run must not create the staging directory"


# ------------------------------------------------------------ never mutates ---


def test_the_source_is_never_modified(source, dest):
    """RAW DATA IS IMMUTABLE: staging is a copy, and --delete is not offered."""
    make_capture(source, "cap_tag_RO1")
    before = sorted(p.relative_to(source).as_posix() for p in source.rglob("*"))

    run("--from", str(source), "--to", str(dest))

    after = sorted(p.relative_to(source).as_posix() for p in source.rglob("*"))
    assert before == after
    assert (source / "cap_tag_RO1.zarr/data.mdb").read_bytes() == b"payload"
    assert "--delete" not in SCRIPT.read_text().replace(
        "--delete is deliberately not offered", ""
    ).replace("No --delete", "")


# ------------------------------------------------------------------- usage ---


def test_missing_arguments_fail_with_usage(source):
    result = run("--from", str(source))
    assert result.returncode != 0
    assert "--to is required" in result.stderr


def test_a_missing_source_directory_is_an_error(tmp_path):
    result = run("--from", str(tmp_path / "nope"), "--to", str(tmp_path / "out"))
    assert result.returncode != 0
    assert "no such source directory" in result.stderr


def test_help_explains_the_family_rule():
    result = run("--help")
    assert result.returncode == 0
    for member in (".zarr", ".yaml", ".log"):
        assert member in result.stdout, member
    assert "set -uo" not in result.stdout, "usage text ran past the header comment"


def test_help_says_only_the_rx_sidecar_is_read():
    """Nobody knew this on 2026-08-04; an offline TX rover blocks nothing."""
    assert "only the RX sidecar" in run("--help").stdout


def test_a_capture_without_a_yaml_is_described_as_tx_usable(source, dest):
    """Over-claiming would send an operator hunting for a TX sidecar."""
    make_capture(source, "cap_tag_RO2", members=("zarr", "log"))

    result = run("--from", str(source), "--to", str(dest))

    assert result.returncode != 0
    assert "TX (emitter)" in result.stdout

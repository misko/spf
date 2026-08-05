"""The per-capture .log sidecar must say what run produced the store beside it.

Regression suite for the zero-byte sidecar: every rover .log was created and
never written because importing spf.mavlink.mavlink_controller calls
logging.basicConfig() at import time, which turned the collector's own
basicConfig into a no-op and sent the run's log to a relative logs.log.
"""

import glob
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import spf
from spf.capture_log import (
    CAPTURE_LOG_HEADER_END,
    CAPTURE_LOG_HEADER_START,
    CONFIG_SOURCE_CANONICAL,
    CONFIG_SOURCE_ENV,
    capture_provenance_lines,
    configure_capture_logging,
    resolve_config_source,
    resolve_rover_id,
    write_capture_provenance,
)

root_dir = os.path.dirname(os.path.dirname(spf.__file__))


def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    return env


def parse_header(text):
    fields = {}
    for line in text.splitlines():
        if line in (CAPTURE_LOG_HEADER_START, CAPTURE_LOG_HEADER_END):
            continue
        if not line.startswith("# "):
            break
        key, _, value = line[2:].partition(": ")
        fields[key] = value
    return fields


@pytest.fixture
def isolated_logging():
    """Restore root logging state; these tests deliberately reconfigure it."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield root
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            handler.close()
    root.handlers = saved_handlers
    root.setLevel(saved_level)


def test_provenance_lines_carry_invocation_and_both_timestamps(tmp_path):
    lines = capture_provenance_lines(
        log_path=tmp_path / "capture.log",
        config_path=tmp_path / "rover4_production_v7.yaml",
        tag="RO4",
        argv=["spf/mavlink_radio_collection.py", "--tag", "RO4"],
        run_started_at=1722790000.0,
    )
    fields = parse_header("\n".join(lines))

    assert lines[0] == CAPTURE_LOG_HEADER_START
    assert lines[-1] == CAPTURE_LOG_HEADER_END
    assert "--tag RO4" in fields["argv"]
    assert fields["tag"] == "RO4"
    assert fields["rover_id"] == "4"
    assert fields["hostname"]
    assert fields["git_commit"]
    # Both clocks, explicitly labelled, and the UTC one unambiguous.
    assert fields["timestamp_utc"].endswith("Z")
    assert fields["timestamp_local"]
    assert fields["timestamp_local_utc_offset"]
    assert fields["timestamp_unix"].startswith("1722790000")
    # The filename stamp is local time; say so, since rovers 1-4 have disagreed
    # about timezone by eight hours and a calendar day.
    assert fields["filename_timestamp_is_local_time"] == "true"


def test_config_path_is_recorded_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lines = capture_provenance_lines(
        log_path="capture.log", config_path="relative_config.yaml", tag="RO1"
    )
    fields = parse_header("\n".join(lines))
    assert Path(fields["capture_config_path"]).is_absolute()


def test_config_source_distinguishes_env_override_from_resolver(tmp_path):
    config = tmp_path / "qualified.yaml"
    config.write_text("{}\n")
    other = tmp_path / "canonical.yaml"
    other.write_text("{}\n")

    assert (
        resolve_config_source(config, env={CONFIG_SOURCE_ENV: str(config)})
        == CONFIG_SOURCE_ENV
    )
    # An override that is set but points somewhere else did not source this run.
    assert (
        resolve_config_source(other, env={CONFIG_SOURCE_ENV: str(config)})
        == CONFIG_SOURCE_CANONICAL
    )
    assert resolve_config_source(other, env={}) == CONFIG_SOURCE_CANONICAL


def test_rover_id_falls_back_env_then_tag_then_file(tmp_path):
    id_file = tmp_path / "rover_id"
    id_file.write_text("2\n")
    assert resolve_rover_id("RO4", env={"SPF_ROVER_ID": "3"}, rover_id_file=id_file) == "3"
    assert resolve_rover_id("RO4", env={}, rover_id_file=id_file) == "4"
    assert (
        resolve_rover_id("BOOT_DIRECT_V7_RO1", env={}, rover_id_file=id_file) == "1"
    )
    assert resolve_rover_id("", env={}, rover_id_file=id_file) == "2"
    assert resolve_rover_id("", env={}, rover_id_file=tmp_path / "missing") == "unknown"


def test_write_provenance_appends_and_never_truncates(tmp_path):
    log_path = tmp_path / "capture.log.tmp"
    log_path.write_text("pre-existing\n")
    write_capture_provenance(log_path, config_path=None, tag="RO3")
    text = log_path.read_text()
    assert text.startswith("pre-existing\n")
    assert CAPTURE_LOG_HEADER_START in text


def test_logging_reaches_sidecar_even_after_an_earlier_basicconfig(
    tmp_path, isolated_logging
):
    """The exact defect: a prior basicConfig used to silently win.

    spf.mavlink.mavlink_controller installs a root handler at import time.
    Without force=True the capture's handlers are never installed and the
    sidecar stays at zero bytes.
    """
    hijack = tmp_path / "logs.log"
    logging.basicConfig(filename=str(hijack), level=logging.DEBUG, force=True)

    log_path = tmp_path / "capture.log.tmp"
    absolute = configure_capture_logging(
        log_path, level="INFO", stream=False, config_path=None, tag="RO4"
    )
    logging.info("collector line that must land in the sidecar")
    logging.shutdown()

    assert absolute.is_absolute()
    text = log_path.read_text()
    assert text
    assert CAPTURE_LOG_HEADER_START in text
    assert "collector line that must land in the sidecar" in text
    assert "collector line that must land in the sidecar" not in hijack.read_text()


def test_sidecar_path_is_absolute_regardless_of_working_directory(
    tmp_path, monkeypatch, isolated_logging
):
    """A relative destination follows the cwd; that is how logs.log escapes."""
    target_dir = tmp_path / "captures"
    target_dir.mkdir()
    monkeypatch.chdir(target_dir)
    absolute = configure_capture_logging(
        "capture.log.tmp", level="INFO", stream=False, config_path=None, tag="RO4"
    )
    monkeypatch.chdir(tmp_path)
    logging.info("written after the process changed directory")
    logging.shutdown()

    assert absolute == (target_dir / "capture.log.tmp").absolute()
    assert "written after the process changed directory" in absolute.read_text()
    assert not (tmp_path / "capture.log.tmp").exists()


def test_fake_drone_capture_writes_a_populated_log_sidecar():
    """End-to-end, no hardware: run the collector and read its sidecar."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        argv = [
            sys.executable,
            f"{root_dir}/spf/mavlink_radio_collection.py",
            "--fake-drone",
            "--exit",
            "-c",
            f"{root_dir}/tests/test_config.yaml",
            "-m",
            f"{root_dir}/tests/test_device_mapping",
            "-r",
            "center",
            "-n",
            "5",
            "--tag",
            "RO4",
            "--temp",
            tmpdirname,
        ]
        subprocess.check_output(
            argv,
            timeout=180,
            env=get_env(),
            stderr=subprocess.STDOUT,
            cwd=tmpdirname,
        )

        log_fns = glob.glob(f"{tmpdirname}/rover_*.log")
        assert len(log_fns) == 1
        text = Path(log_fns[0]).read_text()
        assert text.strip(), "capture .log sidecar is empty"

        fields = parse_header(text)
        assert "mavlink_radio_collection.py" in fields["argv"]
        assert "--tag RO4" in fields["argv"]
        assert fields["capture_config_path"].endswith("tests/test_config.yaml")
        assert fields["capture_config_source"]
        assert fields["rover_id"] == "4"
        assert fields["tag"] == "RO4"
        assert fields["hostname"]
        assert fields["git_commit"]
        assert fields["timestamp_local"]
        assert fields["timestamp_utc"].endswith("Z")
        assert Path(fields["log_sidecar_final"]).name == Path(log_fns[0]).name

        # The collector's own output, not just the header.
        assert "MavRadioCollection" in text
        # And it did not leak into a cwd-relative logs.log.
        stray = Path(tmpdirname) / "logs.log"
        assert not stray.exists() or stray.stat().st_size == 0

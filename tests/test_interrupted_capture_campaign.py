import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = (
    REPO_ROOT
    / "data_collection/rover/rover_v3.1/run_interrupted_capture_campaign.sh"
)


def _executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    ("pytest_status", "after_line", "expected_status", "kernel_usb_error"),
    (
        (7, "post-case diagnostic marker", 7, 0),
        (0, "USB disconnect, simulated", 1, 1),
    ),
)
def test_case_preserves_post_case_kernel_evidence(
    tmp_path, pytest_status, after_line, expected_status, kernel_usb_error
):
    config = tmp_path / "capture.yaml"
    config.write_text("data-version: 7\n")
    mapping = tmp_path / "device_mapping"
    mapping.write_text("2 46\n")
    output = tmp_path / "output"
    dmesg_state = tmp_path / "dmesg-count"

    fake_python = _executable(
        tmp_path / "python",
        f"""#!/usr/bin/env python3
import os
import sys

if sys.argv[1:3] == [\"-m\", \"spf.scripts.rover_capture_config\"]:
    values = [\"\", {str(config)!r}, \"\", \"center\", \"\", \"1\"] + [\"\"] * 9
    sys.stdout.buffer.write((\"\\0\".join(values) + \"\\0\").encode())
elif sys.argv[1:3] == [\"-m\", \"pytest\"]:
    raise SystemExit(int(os.environ[\"FAKE_PYTEST_STATUS\"]))
raise SystemExit(99)
""",
    )
    fake_prepare = _executable(
        tmp_path / "prepare.sh",
        """#!/usr/bin/env bash
set -eu
printf '{\"ready\": true}\n' >"$SPF_DIRECT_USB_READY_FILE"
""",
    )
    fake_sudo = _executable(
        tmp_path / "sudo",
        """#!/usr/bin/env bash
exec "$@"
""",
    )
    fake_dmesg = _executable(
        tmp_path / "dmesg",
        """#!/usr/bin/env bash
set -eu
count=0
if [[ -f "$FAKE_DMESG_STATE" ]]; then
    count="$(cat "$FAKE_DMESG_STATE")"
fi
count=$((count + 1))
printf '%s\n' "$count" >"$FAKE_DMESG_STATE"
printf 'baseline\n'
if (( count > 1 )); then
    printf '%s\n' "$FAKE_DMESG_AFTER"
fi
""",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "SPF_PYTHON": str(fake_python),
            "SPF_ROVER_ID": "2",
            "SPF_CAPTURE_CONFIG": str(config),
            "SPF_INTERRUPT_OUTPUT_ROOT": str(output),
            "SPF_INTERRUPT_CASES": "sigkill:40",
            "SPF_INTERRUPT_CLEAN_RECORDS": "1",
            "SPF_PREPARE_DIRECT_USB_BOOT": str(fake_prepare),
            "SPF_DEVICE_MAPPING": str(mapping),
            "SPF_DMESG_BIN": str(fake_dmesg),
            "SPF_SUDO": str(fake_sudo),
            "FAKE_DMESG_STATE": str(dmesg_state),
            "FAKE_DMESG_AFTER": after_line,
            "FAKE_PYTEST_STATUS": str(pytest_status),
        }
    )

    result = subprocess.run(
        [str(CAMPAIGN)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )

    assert result.returncode == expected_status, result.stdout
    run_roots = list(output.iterdir())
    assert len(run_roots) == 1
    case = run_roots[0] / "case-01-sigkill-40"
    assert (case / "dmesg-before.txt").read_text() == "baseline\n"
    assert (case / "dmesg-after.txt").read_text() == (
        f"baseline\n{after_line}\n"
    )
    assert (case / "dmesg-delta.txt").read_text() == f"{after_line}\n"
    assert (case / "case-status.env").read_text() == (
        f"pytest_status={pytest_status}\n"
        "dmesg_status=0\n"
        f"kernel_usb_error={kernel_usb_error}\n"
    )
    assert not (run_roots[0] / "PASS").exists()

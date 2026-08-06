import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"


def test_every_boot_syncs_the_clock_from_gps_before_anything_is_named():
    """The Pi has no RTC, so a stale restored clock names captures wrongly.

    A former `system_clock_is_plausible` guard skipped the boot sync whenever
    the clock read later than 2025-01-01 -- which a clock restored to four hours
    ago clears by nineteen months. 19 of 47 finalised campaign captures were
    misdated as a result, by up to 4h28m. The guard must not come back.
    """
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    assert "system_clock_is_plausible() {" not in launcher, (
        "the plausibility guard is what caused 19 misdated captures; "
        "every boot must actually ask the GPS"
    )
    # It may be NAMED in a comment recording why it is gone -- that history is
    # worth keeping -- but it must never be invoked again.
    invoked = [
        line
        for line in launcher.splitlines()
        if "system_clock_is_plausible" in line and not line.lstrip().startswith("#")
    ]
    assert not invoked, f"the removed guard is still invoked: {invoked}"
    assert "sync_gps_time boot" in launcher
    assert "sync_gps_time capture" in launcher

    sync_body = launcher.split("sync_gps_time() {", 1)[1].split("\n}\n", 1)[0]
    # No early return before the sync is attempted: the boot path must reach it.
    assert "return 0" not in sync_body.split("for ((", 1)[0]
    # Naming-critical phases get multiple bounded attempts; the opportunistic
    # post-capture sync gets one.
    assert 'gps_time_sync_is_naming_critical "$phase"' in sync_body
    assert 'attempts="$GPS_TIME_SYNC_ATTEMPTS"' in sync_body

    once_body = launcher.split("gps_time_sync_once() {", 1)[1].split("\n}\n", 1)[0]
    assert 'timeout "${SPF_GPS_TIME_TIMEOUT:-180}"' in once_body
    assert 'sudo date -s "$gps_time"' in once_body
    # Never set the clock to epoch 0 on a fix-without-UTC.
    assert '[[ "$gps_time" == 1970-* ]]' in once_body


def test_gps_time_sync_failure_cannot_abort_the_boot_under_set_e():
    """sync_gps_time now returns non-zero; a bare call would kill the launcher."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    assert "set -euo pipefail" in launcher
    assert "sync_gps_time boot || " in launcher
    assert "sync_gps_time capture || true" in launcher


def test_an_unverified_clock_is_announced_loudly():
    """Silent failure here is what let misdated captures go unnoticed for a week."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    sync_body = launcher.split("sync_gps_time() {", 1)[1].split("\n}\n", 1)[0]

    assert "WARNING: no GPS UTC after" in sync_body
    assert "UNVERIFIED" in sync_body
    assert "never on the filename" in sync_body


def test_gps_time_wait_notifies_once_instead_of_beeping_each_poll():
    source = (REPO_ROOT / "spf/mavlink/mavlink_controller.py").read_text()
    get_time_body = source.split("if args.get_time is not None:", 1)[1].split(
        "\n    if (", 1
    )[0]

    assert get_time_body.count('drone.buzzer(tones["gps-time"])') == 1


def test_capture_failure_is_durable_and_plays_three_operator_tones():
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    run_capture = launcher.split("run_capture() {", 1)[1].split("\n}\n", 1)[0]
    notify = launcher.split("notify_capture_failure() {", 1)[1].split("\n}\n", 1)[0]

    assert '--status-file "$CAPTURE_STATUS_FILE"' in run_capture
    assert "spf.capture_status mark-failed" in run_capture
    assert "for _attempt in 1 2 3" in notify
    assert '"$MAVLINK_CONTROLLER" --buzzer failure' in notify
    assert 'return "$capture_status"' in run_capture


def test_launcher_attempts_hold_before_failure_tones_or_radio_retry():
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    main_loop = launcher.split("while true; do", 1)[1].split("\n    done", 1)[0]

    assert main_loop.index("ensure_vehicle_hold_after_capture_failure") < (
        main_loop.index("notify_capture_failure")
    )
    assert main_loop.index("notify_capture_failure") < main_loop.index(
        "revalidate_radios_after_capture_failure"
    )


def test_capture_runs_with_an_independent_bounded_watchdog():
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    run_capture = launcher.split("run_capture() {", 1)[1].split("\n}\n", 1)[0]

    assert "spf.capture_watchdog monitor" in run_capture
    assert '--pid "$capture_pid"' in run_capture
    assert '--expected-plutos "$expected_radios"' in run_capture
    assert '--status-file "$CAPTURE_STATUS_FILE"' in run_capture
    assert 'wait "$capture_pid"' in run_capture
    assert 'wait "$watchdog_pid"' in run_capture


def test_failed_capture_can_only_restart_as_a_new_attested_artifact():
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    revalidate = launcher.split("revalidate_radios_after_capture_failure() {", 1)[
        1
    ].split("\n}\n", 1)[0]
    main_loop = launcher.split("while true; do", 1)[1].split("\n    done", 1)[0]

    assert "SPF_CAPTURE_RESTART_ATTEMPTS" in launcher
    assert 'device_mapping.sh" >"$mapping_candidate"' in revalidate
    assert "spf.scripts.pluto_ready_manifest" in revalidate
    assert 'refresh "${manifest_args[@]}"' in revalidate
    assert 'verify "${manifest_args[@]}"' in revalidate
    assert "if run_capture; then" in main_loop
    assert "revalidate_radios_after_capture_failure" in main_loop
    assert "consecutive_capture_failures" in main_loop
    assert "ensure_vehicle_hold_after_capture_failure" in main_loop
    assert "--mode HOLD" in launcher


def test_post_capture_navigation_failure_cannot_relabel_a_complete_zarr():
    source = (REPO_ROOT / "spf/mavlink_radio_collection.py").read_text()
    post_capture = source.split(
        "# Post-capture navigation is operationally important", 1
    )[1]

    assert "try:" in post_capture
    assert "if not drone.move_to_home():" in post_capture
    assert "return-home operation did not reach home" in post_capture
    assert "except BaseException as error:" in post_capture
    assert 'error_source="post_capture_navigation"' in post_capture
    assert "terminate_capture_process(" in post_capture


def test_heavy_collection_imports_are_deferred_until_after_drone_ready_wait():
    path = REPO_ROOT / "spf/mavlink_radio_collection.py"
    source = path.read_text()
    module = ast.parse(source)
    deferred_modules = {
        "spf.data_collector",
        "spf.dataset.spf_dataset",
        "spf.dataset.spf_nn_dataset_wrapper",
        "spf.distance_finder.distance_finder_controller",
        "spf.scripts.train_utils",
    }
    eager_imports = {
        node.module
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert deferred_modules.isdisjoint(eager_imports)
    assert source.index("Drone startup wait for drone ready") < source.index(
        "from spf.data_collector import"
    )


def test_no_capture_is_named_before_the_clock_is_gps_verified():
    """The filename is stamped when the capture process starts, not when it ends.

    Syncing only AFTER a capture left exactly one capture per session misnamed
    whenever the boot sync had failed: capture 1 stamped its name from the stale
    clock, then completed, then fixed the clock for capture 2 onward.
    """
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    main_loop = launcher.split("while true; do", 1)[1].split("\n    done", 1)[0]

    assert "ensure_clock_verified_before_capture" in main_loop
    assert main_loop.index("ensure_clock_verified_before_capture") < main_loop.index(
        "if run_capture; then"
    ), "the clock must be verified BEFORE the capture names its artifact"


def test_pre_capture_sync_gets_the_full_retry_budget():
    """boot and pre-capture both name artifacts; the post-capture sync does not."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    critical = launcher.split("gps_time_sync_is_naming_critical() {", 1)[1].split(
        "\n}\n", 1
    )[0]

    assert '"$1" == "boot"' in critical
    assert '"$1" == "pre-capture"' in critical
    assert "sync_gps_time pre-capture" in launcher
    assert "sync_gps_time capture || true" in launcher


def test_a_verified_clock_is_not_resynced_before_every_capture():
    """Once set from GPS the clock is good for the session; re-syncing before every capture
    would add a MAVLink round trip and an operator tone to every run."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    ensure = launcher.split("ensure_clock_verified_before_capture() {", 1)[1].split(
        "\n}\n", 1
    )[0]

    assert '[[ "$CLOCK_VERIFIED_FROM_GPS" -eq 1 ]]' in ensure
    assert "return 0" in ensure.split('CLOCK_VERIFIED_FROM_GPS" -eq 1', 1)[1]
    # the flag is only ever set by a successful sync
    sync_body = launcher.split("sync_gps_time() {", 1)[1].split("\n}\n", 1)[0]
    assert "CLOCK_VERIFIED_FROM_GPS=1" in sync_body


def test_an_unverified_capture_says_the_store_is_still_trustworthy():
    """Operators must not conclude the DATA is bad -- only the filename is."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    ensure = launcher.split("ensure_clock_verified_before_capture() {", 1)[1].split(
        "\n}\n", 1
    )[0]

    assert "UNVERIFIED clock" in ensure
    assert "gps_timestamp inside the store is still correct" in ensure

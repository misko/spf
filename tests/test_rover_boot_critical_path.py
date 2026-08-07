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
    assert 'timeout "$GPS_TIME_TIMEOUT_S"' in once_body
    # The epoch-0 guard and the clock-setting call are asserted in detail by
    # test_the_clock_is_validated_as_an_epoch_not_by_a_1970_prefix.


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


def test_print_plan_variables_are_all_defined_before_it_runs():
    """`--print-plan` exits at the top of the script, under `set -u`.

    Defining a tunable next to the function that uses it, rather than with the
    other env-derived knobs, makes `--print-plan` die with "unbound variable" on
    a real rover -- and it cannot be caught locally, because drone_run.sh exits
    earlier here on the missing /home/pi paths. Static check instead.
    """
    import re

    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    body = launcher.split("print_plan() {", 1)[1].split("\n}\n", 1)[0]
    referenced = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", body)))
    assert referenced, "print_plan should reference variables"

    before_call = launcher[: launcher.index("--print-plan)")]
    unbound = [
        name
        for name in referenced
        if not re.search(
            rf"^\s*(readonly\s+)?{name}=|mapfile.*\b{name}\b", before_call, re.M
        )
    ]
    assert not unbound, (
        f"--print-plan would die on unbound variable(s): {unbound}. "
        "Define them with the other env-derived knobs near the top."
    )


def test_gps_clock_knobs_are_validated_and_visible():
    """SPF_GPS_TIME_SYNC_ATTEMPTS=0 or SPF_GPS_TIME_TIMEOUT=0 would silently
    reinstate the defect: a zero-iteration retry loop, and GNU `timeout 0`
    disables the timeout rather than meaning "do not wait"."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    assert '[[ "$GPS_TIME_SYNC_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] ||' in launcher
    assert '[[ "$GPS_TIME_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] ||' in launcher
    plan = launcher.split("print_plan() {", 1)[1].split("\n}\n", 1)[0]
    assert "gps_time_sync_attempts=" in plan
    assert "gps_time_timeout_s=" in plan


def test_the_clock_is_validated_as_an_epoch_not_by_a_1970_prefix():
    """datetime.fromtimestamp(0) is naive LOCAL time: "1970-01-01 01:00:00" in
    Europe/London but "1969-12-31 16:00:00" west of UTC, where a "1970-*" prefix
    test misses entirely and `date -s` really does set the clock to epoch 0.
    And `date -d ""` does NOT fail -- it parses to today at midnight."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    once = launcher.split("gps_time_sync_once() {", 1)[1].split("\n}\n", 1)[0]

    assert '[[ "$gps_time" == 1970-* ]]' not in once, "prefix test is timezone-broken"
    assert '[[ -z "${gps_time//[[:space:]]/}" ]]' in once, "empty file must be rejected"
    assert 'date -d "$gps_time" +%s' in once, "parse to an epoch"
    assert '"$epoch" -lt 1735689600' in once, "floor the epoch"
    assert 'sudo -n date -s "@$epoch"' in once, "set from the validated epoch, with sudo -n"


def test_get_time_waits_for_utc_not_merely_for_a_fix():
    """The old OR-exit returned on a 3D fix while gps_time was still 0, writing
    an epoch-0 timestamp; and its literal did not match DGPS/RTK fix strings."""
    source = (REPO_ROOT / "spf/mavlink/mavlink_controller.py").read_text()
    body = source.split("if args.get_time is not None:", 1)[1].split("\n    if (", 1)[0]

    assert "while drone.gps_time == 0:" in body
    assert 'gps_fix_type != "GPS_FIX_TYPE_3D_FIX"' not in body
    # The file must be opened only once a real value exists: opening it before
    # the wait meant a timeout-killed attempt truncated it to zero bytes.
    assert body.index("while drone.gps_time == 0:") < body.index("open(args.get_time")


def test_removed_flags_fail_closed_instead_of_being_ignored():
    """A rover parked with SPF_BOOT_VALIDATE_ONLY=1 must not silently start moving.

    Rovers pull origin/main at boot, so ignoring a stale line would put a
    deliberately-parked rover into the motion-capable mission loop unattended.
    """
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    assert 'if [[ -n "${SPF_BOOT_VALIDATE_ONLY:-}" ]]; then' in launcher
    assert 'if [[ -n "${SPF_RUN_ONCE:-}" ]]; then' in launcher
    assert "MOTION-CAPABLE" in launcher

    # Only a flag that was actually ENGAGED may refuse to boot. "=0" is the
    # harmless production default present on every rover, and dying on it would
    # take the whole fleet down at the next boot -- unattended, since rovers pull
    # origin/main at boot.
    guard = launcher.split('if [[ -n "${SPF_BOOT_VALIDATE_ONLY:-}" ]]; then', 1)[1]
    guard = guard.split("\nfi\n", 1)[0]
    assert 'is_true "${SPF_BOOT_VALIDATE_ONLY}"' in guard, "=0 must not be fatal"
    assert "die " in guard, "=1 must be fatal"
    assert "NOTE:" in guard, "=0 should still tell the operator to delete the line"

    # The behaviour itself is gone: no branch, no CLI flag, no plan key.
    assert 'is_true "$BOOT_VALIDATE_ONLY"' not in launcher
    assert 'is_true "$RUN_ONCE"' not in launcher
    assert "--boot-validate-only)" not in launcher
    assert "--once)" not in launcher
    plan = launcher.split("print_plan() {", 1)[1].split("\n}\n", 1)[0]
    assert "boot_validate_only=" not in plan
    assert "run_once=" not in plan


def test_the_stall_knobs_reach_the_collector_from_the_boot_path():
    """run_capture() built a fixed argument list that omitted them entirely.

    mavlink_radio_collection.py has accepted --stall-detect-seconds,
    --stall-manual-seconds and --stall-progress-radius-m all along, but nothing
    on the boot path could pass one. Retuning the watchdog after a field session
    meant editing this script on four rovers.
    """
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    for knob in (
        "SPF_STALL_DETECT_SECONDS",
        "SPF_STALL_MANUAL_SECONDS",
        "SPF_STALL_PROGRESS_RADIUS_M",
    ):
        assert knob in launcher, f"{knob} is not readable from the environment"

    capture = launcher.split("run_capture() {", 1)[1].split("\n}\n", 1)[0]
    assert 'stall_args[@]+"${stall_args[@]}"' in capture, (
        "the stall flags must reach the collector, and the +expansion is "
        "required: an empty array under `set -u` is an unbound variable"
    )

    for flag in (
        "--stall-detect-seconds",
        "--stall-manual-seconds",
        "--stall-progress-radius-m",
    ):
        assert flag in launcher


def test_an_unset_stall_knob_passes_no_flag_at_all():
    """Defaults keep exactly one home -- the collector's argparse.

    Mirroring the numbers here is how the boot path and the `rover` CLI drift
    apart; the empty default means an unset knob is genuinely absent rather
    than re-specified.
    """
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    assert 'STALL_DETECT_SECONDS="${SPF_STALL_DETECT_SECONDS:-}"' in launcher
    assert 'STALL_MANUAL_SECONDS="${SPF_STALL_MANUAL_SECONDS:-}"' in launcher
    assert 'STALL_PROGRESS_RADIUS_M="${SPF_STALL_PROGRESS_RADIUS_M:-}"' in launcher
    assert "stall_args=()" in launcher


def test_a_zero_stall_knob_is_refused():
    """`--stall-detect-seconds 0` stalls every waypoint instantly, and the rover
    spends the session escaping jams that are not there. Absent is safe; zero
    is not."""
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    block = launcher.split("for stall_knob in", 1)[1].split("\ndone\n", 1)[0]

    assert '"$stall_knob_value" =~ [1-9]' in block, (
        "a decimal of all zeroes (0, 0.0, 0.00) must be refused"
    )
    assert "must be a positive number" in block
    # `[[ ... ]] && continue` returns nonzero for every knob that IS set, which
    # under `set -e` aborts the boot -- the repeat gotcha in this script.
    assert '[[ -z "$stall_knob_value" ]] && continue' not in block
    assert 'if [[ -n "$stall_knob_value" ]]; then' in block


def test_the_stall_knobs_are_visible_in_the_plan():
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    plan = launcher.split("print_plan() {", 1)[1].split("\n}\n", 1)[0]

    assert "stall_detect_seconds=" in plan
    assert "stall_manual_seconds=" in plan
    assert "stall_progress_radius_m=" in plan

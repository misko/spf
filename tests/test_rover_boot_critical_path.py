import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"


def test_plausible_clock_defers_blocking_gps_time_sync_at_boot():
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    assert "system_clock_is_plausible" in launcher
    assert "sync_gps_time boot" in launcher
    assert "sync_gps_time capture" in launcher
    sync_body = launcher.split("sync_gps_time() {", 1)[1].split("\n}\n", 1)[0]
    assert '[[ "$phase" == "boot" ]] && system_clock_is_plausible' in sync_body
    assert sync_body.index("system_clock_is_plausible") < sync_body.index(
        'timeout "${SPF_GPS_TIME_TIMEOUT:-180}"'
    )


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

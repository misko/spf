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

import glob
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

import docker
import spf.mavlink.mavlink_controller
from spf import mavlink_radio_collection
from spf.dataset.v4_data import v4rx_f64_keys
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

root_dir = os.path.dirname(os.path.dirname(spf.__file__))

simulator_speedup = 5

"""
docker run --rm -it -p 14590-14595:14590-14595 ardupilot_spf /ardupilot/Tools/autotest/sim_vehicle.py \
   -l 37.76509485,-122.40940127,0,0 -v rover -f rover-skid \
    --out tcpin:0.0.0.0:14590  --out tcpin:0.0.0.0:14591 -S 1 
    """


@pytest.fixture(scope="session")
def adrupilot_simulator():
    client = docker.from_env()
    container = client.containers.run(
        # "csmisko/ardupilotspf:latest",
        "ghcr.io/misko/ardupilotspf:v0.2",
        f"/ardupilot/Tools/autotest/sim_vehicle.py  -l 37.76509485,-122.40940127,0,0 \
            -v rover -f rover-skid --out tcpin:0.0.0.0:14590  --out tcpin:0.0.0.0:14591 -S {simulator_speedup}",
        stdin_open=True,
        ports={
            "14590/tcp": ("127.0.0.1", 14590),
            "14591/tcp": ("127.0.0.1", 14591),
        },
        detach=True,
        remove=True,
        auto_remove=True,
    )
    try:
        output = container.attach(stdout=True, stream=True, logs=True)
        online = False
        for line in output:
            if "Detected vehicle" in line.decode():
                online = True
                break

        if not online:
            raise ValueError

        yield
    finally:
        container.stop()


def mavlink_controller_base_command(port=14591):
    return f"python3 {spf.mavlink.mavlink_controller.__file__} --ip 127.0.0.1 --port {port} --proto tcp"


def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    return env


def get_gps_time():
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "/gps_time"
        subprocess.check_output(
            f"{mavlink_controller_base_command()} --get-time {file_name}",
            timeout=30,
            shell=True,
            env=get_env(),
        )
        assert os.path.isfile(file_name)
        with open(file_name, "r") as f:
            return f.readlines()


def get_time_since_boot():
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "/gps_time"
        subprocess.check_output(
            f"{mavlink_controller_base_command()} --time-since-boot {file_name}",
            timeout=30,
            shell=True,
            env=get_env(),
        )
        assert os.path.isfile(file_name)
        with open(file_name, "r") as f:
            return f.readlines()


def buzzer(tone):
    subprocess.check_output(
        f"{mavlink_controller_base_command()} --buzzer {tone}",
        timeout=30,
        shell=True,
        env=get_env(),
    )


def set_mode(mode, port=14591):
    print("SET MODE", f"{mavlink_controller_base_command(port)} --mode {mode}")
    subprocess.check_output(
        f"{mavlink_controller_base_command(port)} --mode {mode}",
        timeout=30,
        shell=True,
        env=get_env(),
    )


def test_gps_time(adrupilot_simulator):
    assert len(get_gps_time()) != 0


def test_time_since_boot(adrupilot_simulator):
    assert len(get_time_since_boot()) != 0


def test_reboot(adrupilot_simulator):
    time1 = float(get_time_since_boot()[0]) / simulator_speedup
    start_time = time.time()
    time.sleep(1)
    end_time = time.time()
    time.sleep(1)
    time2 = float(get_time_since_boot()[0]) / simulator_speedup
    assert (time2 - time1) > (end_time - start_time)
    assert (end_time - start_time) - (time2 - time1) < 20

    start_time = time.time()
    subprocess.check_output(
        f"{mavlink_controller_base_command()} --reboot",
        timeout=30,
        shell=True,
        stderr=subprocess.STDOUT,
        env=get_env(),
    )
    time_since_boot = float(get_time_since_boot()[0]) / simulator_speedup
    time.sleep(0.5)  # takes some time to write to disk
    end_time = time.time()
    assert (end_time - start_time) > time_since_boot


def generate_parameters_file(rover_id, file_name):
    subprocess.check_output(
        f"cat {root_dir}/data_collection/rover/rover_v3.1/rover3_base_parameters.params \
              | sed 's/__ROVER_ID__/{rover_id}/g' > {file_name}",
        timeout=30,
        shell=True,
        stderr=subprocess.STDOUT,
        env=get_env(),
    )


def load_params(file_name):
    subprocess.check_output(
        f"{mavlink_controller_base_command()} --load-params {file_name}",
        timeout=180,
        shell=True,
        stderr=subprocess.STDOUT,
        env=get_env(),
    )


def diff_params(file_name):
    print(f"{mavlink_controller_base_command()} --diff-params {file_name}")
    subprocess.check_output(
        f"{mavlink_controller_base_command()} --diff-params {file_name}",
        timeout=180,
        shell=True,
        stderr=subprocess.STDOUT,
        env=get_env(),
    )


def test_load_and_diff_params(adrupilot_simulator):
    with tempfile.TemporaryDirectory() as tmpdirname:
        param_file_nameA = tmpdirname + "/this_droneA.params"
        generate_parameters_file(5, param_file_nameA)
        param_file_nameB = tmpdirname + "/this_droneB.params"
        generate_parameters_file(6, param_file_nameB)

        load_params(param_file_nameA)
        diff_params(param_file_nameA)
        with pytest.raises(subprocess.CalledProcessError):
            diff_params(param_file_nameB)
        load_params(param_file_nameB)
        diff_params(param_file_nameB)


def test_buzzer(adrupilot_simulator):
    buzzer("gps-time")
    buzzer("check-diff")
    buzzer("git")
    buzzer("planner")
    buzzer("ready")


def test_mode(adrupilot_simulator):
    set_mode("manual")
    set_mode("guided")


def mavlink_radio_collection_base_command():
    return f"python3 {mavlink_radio_collection.__file__} -c {root_dir}/tests/rover_config.yaml -m \
          {root_dir}/tests/device_mapping"


def test_manual_mode_stationary(adrupilot_simulator):
    set_mode("manual")
    with tempfile.TemporaryDirectory() as tmpdirname:
        output = subprocess.check_output(
            f"{mavlink_radio_collection_base_command()}  -r circle --temp {tmpdirname} -s 30",
            timeout=180,
            shell=True,
            env=get_env(),
            stderr=subprocess.STDOUT,
        ).decode()
        assert "MavRadioCollection: Waiting for drone to start moving" in output
        assert "Planner starting to issue move commands" not in output
        assert glob.glob(f"{tmpdirname}/*.zarr.tmp")
        assert glob.glob(f"{tmpdirname}/*.log.tmp")
        assert glob.glob(f"{tmpdirname}/*.yaml.tmp")


def test_guided_mode_moving_and_recording(adrupilot_simulator):
    set_mode("manual")
    with tempfile.TemporaryDirectory() as tmpdirname:
        cmd = (
            f"{mavlink_radio_collection_base_command()}  -r circle --temp {tmpdirname}"
        )
        outputs = []
        with subprocess.Popen(
            cmd,
            shell=True,
            env=get_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,  # Ensures stdout is text rather than bytes
        ) as process:

            # Read each line as it arrives
            for line in process.stdout:
                # Do whatever processing you need on each line
                if "waiting for rover to move into guided mode..." in line:
                    print("SET GUIDED...")
                    set_mode("guided", port=14590)  # other port is busy!
                    print("SET GUIDED")
                print(line)
                outputs.append(line)

            # After the loop ends, the process should have terminated.
            returncode = process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd)

        assert "Planner starting to issue move commands" in "\n".join(outputs)
        assert glob.glob(f"{tmpdirname}/*.zarr")
        assert glob.glob(f"{tmpdirname}/*.log")
        assert glob.glob(f"{tmpdirname}/*.yaml")

        # load output and make sure entries are not obiously wrong
        zarr_fn = glob.glob(f"{tmpdirname}/*.zarr")[0]
        z = zarr_open_from_lmdb_store(zarr_fn)
        keys_with_nans = []
        for key in v4rx_f64_keys:
            if not np.isfinite(z["receivers/r0"][key]).all():
                keys_with_nans.append(key)
        assert len(keys_with_nans) == 0

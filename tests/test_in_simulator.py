import glob
import os
import subprocess
import sys
import tempfile
import time

import docker
import pytest

import spf.mavlink.mavlink_controller
from spf import mavlink_radio_collection

root_dir = os.path.dirname(os.path.dirname(spf.__file__))


@pytest.fixture(scope="session")
def adrupilot_simulator():
    client = docker.from_env()
    container = client.containers.run(
        "csmisko/ardupilotspf:latest",
        "/ardupilot/Tools/autotest/sim_vehicle.py  -l 37.76509485,-122.40940127,0,0 \
            -v rover -f rover-skid --out tcpin:0.0.0.0:14590  --out tcpin:0.0.0.0:14591 -S 5",
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


def mavlink_controller_base_command():
    return f"python3 {spf.mavlink.mavlink_controller.__file__} --ip 127.0.0.1 --port 14591 --proto tcp"


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


def set_mode(mode):
    subprocess.check_output(
        f"{mavlink_controller_base_command()} --mode {mode}",
        timeout=30,
        shell=True,
        env=get_env(),
    )


def test_gps_time(adrupilot_simulator):
    assert len(get_gps_time()) != 0


def test_time_since_boot(adrupilot_simulator):
    assert len(get_time_since_boot()) != 0


def test_reboot(adrupilot_simulator):
    time1 = float(get_time_since_boot()[0])
    start_time = time.time()
    time.sleep(1)
    end_time = time.time()
    time.sleep(1)
    time2 = float(get_time_since_boot()[0])
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
    time_since_boot = float(get_time_since_boot()[0])
    time.sleep(0.5)  # takes some time to write to disk
    end_time = time.time()
    assert (end_time - start_time) > time_since_boot


def generate_parameters_file(rover_id, file_name):
    subprocess.check_output(
        f"cat {root_dir}/data_collection_model_and_results/rover/rover_v3.1/rover3_base_parameters.params  | sed 's/__ROVER_ID__/{rover_id}/g' > {file_name}",
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
    return f"python3 {mavlink_radio_collection.__file__} -c {root_dir}/tests/rover_config.yaml -m {root_dir}/tests/device_mapping --fake-radio"


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
        assert glob.glob(f"{tmpdirname}/*.npy.tmp")
        assert glob.glob(f"{tmpdirname}/*.log.tmp")
        assert glob.glob(f"{tmpdirname}/*.yaml.tmp")


def test_guided_mode_moving_and_recording(adrupilot_simulator):
    set_mode("guided")
    with tempfile.TemporaryDirectory() as tmpdirname:
        output = subprocess.check_output(
            f"{mavlink_radio_collection_base_command()}  -r circle --temp {tmpdirname}",
            timeout=180,
            shell=True,
            env=get_env(),
            stderr=subprocess.STDOUT,
        ).decode()
        assert "Planner starting to issue move commands" in output
        assert glob.glob(f"{tmpdirname}/*.npy")
        assert glob.glob(f"{tmpdirname}/*.log")
        assert glob.glob(f"{tmpdirname}/*.yaml")


# grep  mavlink_radio_collection.log > /dev/null
# if [ $? -ne 0 ]; then
#   echo "  Failed record radio manual mode - start moving"
# fi
# grep "Planner starting to issue move commands" mavlink_radio_collection.log > /dev/null
# if [ $? -eq 0 ]; then
#   echo "  Failed record radio manual mode - start moving"
# fi
# check_dir $tmpdir ".tmp"
# rm ${tmpdir}/*
# rmdir $tmpdir
# # function check_dir () {
#   tmpdir=$1
#   ext=$2
#   if [ ! -f ${tmpdir}/*.log${ext} ]; then
#     echo "  Failed to find log file"
#   fi
#   if [ ! -f ${tmpdir}/*.npy${ext} ]; then
#     echo "  Failed to find npy file"
#   fi
#   if [ ! -f ${tmpdir}/*.yaml${ext} ]; then
#     echo "  Failed to find yaml file"
#   fi
# }


# rover_id=5
# params_root=${repo_root}/data_collection_model_and_results/rover/rover_v3.1/
# cat ${params_root}/rover3_base_parameters.params | sed "s/__ROVER_ID__/${rover_id}/g" > ${repo_root}/tests/this_rover.params

# echo "Test write SYSID 5"
# mavlink_controller --load-params ${repo_root}/tests/this_rover.params
# if [ $? -ne 0 ]; then
#   echo "  Failed write SYSID 5"
# fi

# echo "Test diff with SYSID 5"
# mavlink_controller --diff-params ${repo_root}/tests/this_rover.params
# if [ $? -ne 0 ]; then
#   echo "  Failed diff SYSID 5"
# fi

# rover_id=6
# params_root=${repo_root}/data_collection_model_and_results/rover/rover_v3.1/
# cat ${params_root}/rover3_base_parameters.params | sed "s/__ROVER_ID__/${rover_id}/g" > ${repo_root}/tests/this_rover.params

# echo "Test diff with SYSID 6"
# mavlink_controller --diff-params ${repo_root}/tests/this_rover.params
# if [ $? -eq 0 ]; then
#   echo "  Failed diff SYSID 6"
# fi

# echo "Test write with SYSID 6"
# mavlink_controller --load-params ${repo_root}/tests/this_rover.params
# if [ $? -ne 0 ]; then
#   echo "  Failed write SYSID 6"
# fi

# mavlink_controller --diff-params ${repo_root}/tests/this_rover.params
# echo "Test diff with SYSID 6"
# if [ $? -ne 0 ]; then
#   echo "  Failed diff SYSID 6"
# fi

# echo "Test record radio manual mode"
# tmpdir=`mktemp -d`
# mavlink_radio_collection -r circle --temp ${tmpdir} -s 10
# if [ $? -ne 0 ]; then
#   echo "  Failed record radio manual mode"
# fi
# grep "MavRadioCollection: Waiting for drone to start moving" mavlink_radio_collection.log > /dev/null
# if [ $? -ne 0 ]; then
#   echo "  Failed record radio manual mode - start moving"
# fi
# grep "Planner starting to issue move commands" mavlink_radio_collection.log > /dev/null
# if [ $? -eq 0 ]; then
#   echo "  Failed record radio manual mode - start moving"
# fi
# check_dir $tmpdir ".tmp"
# rm ${tmpdir}/*
# rmdir $tmpdir

# echo "Test fakemode"
# mavlink_controller --mode fakemode
# if [ $? -ne 1 ]; then
#     echo "Failed fakemode mode"
# fi

# echo "Test guided mode"
# mavlink_controller --mode guided
# if [ $? -ne 0  ]; then
#     echo "Failed guided mode"
# fi

# echo "Test record radio guided mode"
# tmpdir=`mktemp -d`
# mavlink_radio_collection -r circle --temp ${tmpdir} -s 30
# if [ $? -ne 0 ]; then
#   echo "  Failed record radio manual mode"
# fi
# grep "Planner starting to issue move commands" mavlink_radio_collection.log > /dev/null
# if [ $? -ne 0 ]; then
#   echo "  Failed record radio manual mode - start moving"
# fi
# check_dir $tmpdir ""
# rm ${tmpdir}/*
# rmdir $tmpdir

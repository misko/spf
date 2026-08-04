import glob
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SimEndpoints:
    """Host ports for the sim's two MAVLink endpoints.

    Keyed by ROLE, never by position. Each `tcpin` is a single-client TCP
    server, so the collector and the commanding client have to land on
    different ones; a positional pair would let an edit silently swap them and
    the symptom would be a timeout rather than an obvious error.

    Docker assigns these, and it does not preserve order or contiguity --
    observed 14590 -> 32769 alongside 14591 -> 32768.
    """

    collect: int  # container 14590 -- data collection
    command: int  # container 14591 -- ground control / commanding


@pytest.fixture(scope="session")
def adrupilot_simulator():
    client = docker.from_env()
    container = client.containers.run(
        "csmisko/ardupilotspf:latest",
        f"/ardupilot/Tools/autotest/sim_vehicle.py  -l 37.76509485,-122.40940127,0,0 \
            -v rover -f rover-skid --out tcpin:0.0.0.0:14590  --out tcpin:0.0.0.0:14591 -S {simulator_speedup}",
        stdin_open=True,
        # Container ports stay 14590/14591 -- sim_vehicle.py bakes them into the
        # --out arguments above. Only the HOST side floats: None asks Docker for
        # a free port. CI and the dev sim share one box (kalman, 192.168.1.141),
        # and pinning the host side made them fight over the same two ports --
        # "Bind for 127.0.0.1:14591 failed: port is already allocated".
        #
        # Docker holds each port from allocation through container life, so
        # there is no window for anything else to take it. Binding :0 ourselves
        # and passing the number on would reintroduce exactly that race, as a
        # rare and therefore flaky failure.
        ports={
            "14590/tcp": ("127.0.0.1", None),
            "14591/tcp": ("127.0.0.1", None),
        },
        detach=True,
        remove=True,
        auto_remove=True,
    )
    try:
        container.reload()  # populates .ports with what Docker actually chose

        def host_port(container_port: int) -> int:
            bindings = container.ports.get(f"{container_port}/tcp")
            if not bindings:
                raise RuntimeError(
                    f"docker published no host port for {container_port}/tcp; "
                    f"got {container.ports!r}"
                )
            return int(bindings[0]["HostPort"])

        endpoints = SimEndpoints(
            collect=host_port(14590), command=host_port(14591)
        )

        output = container.attach(stdout=True, stream=True, logs=True)
        online = False
        for line in output:
            if "Detected vehicle" in line.decode():
                online = True
                break

        if not online:
            raise ValueError

        yield endpoints
    finally:
        container.stop()


def mavlink_controller_base_command(port):
    # `port` is deliberately required. With a default of 14591 a missed call
    # site would still connect -- to whatever owns 14591 on this box, which on
    # the CI host is the developer's own SITL. Silent cross-talk with a live
    # dev vehicle is far worse than a TypeError.
    return f"python3 {spf.mavlink.mavlink_controller.__file__} --ip 127.0.0.1 --port {port} --proto tcp"


def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    return env


def get_gps_time(port):
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "/gps_time"
        subprocess.check_output(
            f"{mavlink_controller_base_command(port)} --get-time {file_name}",
            timeout=30,
            shell=True,
            env=get_env(),
        )
        assert os.path.isfile(file_name)
        with open(file_name, "r") as f:
            return f.readlines()


def get_time_since_boot(port):
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "/gps_time"
        subprocess.check_output(
            f"{mavlink_controller_base_command(port)} --time-since-boot {file_name}",
            timeout=30,
            shell=True,
            env=get_env(),
        )
        assert os.path.isfile(file_name)
        with open(file_name, "r") as f:
            return f.readlines()


def buzzer(port, tone):
    subprocess.check_output(
        f"{mavlink_controller_base_command(port)} --buzzer {tone}",
        timeout=30,
        shell=True,
        env=get_env(),
    )


def set_mode(mode, port, sleep_time=0):
    print("SET MODE", f"{mavlink_controller_base_command(port)} --mode {mode}")
    subprocess.check_output(
        f"{mavlink_controller_base_command(port)} --mode {mode}",
        timeout=30,
        shell=True,
        env=get_env(),
    )
    if sleep_time:
        time.sleep(sleep_time)


def test_gps_time(adrupilot_simulator):
    assert len(get_gps_time(adrupilot_simulator.command)) != 0


def test_time_since_boot(adrupilot_simulator):
    assert len(get_time_since_boot(adrupilot_simulator.command)) != 0


def test_reboot(adrupilot_simulator):
    port = adrupilot_simulator.command
    time1 = float(get_time_since_boot(port)[0]) / simulator_speedup
    start_time = time.time()
    time.sleep(1)
    end_time = time.time()
    time.sleep(1)
    time2 = float(get_time_since_boot(port)[0]) / simulator_speedup
    assert (time2 - time1) > (end_time - start_time)
    assert (end_time - start_time) - (time2 - time1) < 20

    start_time = time.time()
    subprocess.check_output(
        f"{mavlink_controller_base_command(port)} --reboot",
        timeout=30,
        shell=True,
        stderr=subprocess.STDOUT,
        env=get_env(),
    )
    time_since_boot = float(get_time_since_boot(port)[0]) / simulator_speedup
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


def load_params(port, file_name):
    subprocess.check_output(
        f"{mavlink_controller_base_command(port)} --load-params {file_name}",
        timeout=180,
        shell=True,
        stderr=subprocess.STDOUT,
        env=get_env(),
    )


def diff_params(port, file_name):
    print(f"{mavlink_controller_base_command(port)} --diff-params {file_name}")
    subprocess.check_output(
        f"{mavlink_controller_base_command(port)} --diff-params {file_name}",
        timeout=180,
        shell=True,
        stderr=subprocess.STDOUT,
        env=get_env(),
    )


def test_load_and_diff_params(adrupilot_simulator):
    port = adrupilot_simulator.command
    with tempfile.TemporaryDirectory() as tmpdirname:
        param_file_nameA = tmpdirname + "/this_droneA.params"
        generate_parameters_file(5, param_file_nameA)
        param_file_nameB = tmpdirname + "/this_droneB.params"
        generate_parameters_file(6, param_file_nameB)

        load_params(port, param_file_nameA)
        diff_params(port, param_file_nameA)
        with pytest.raises(subprocess.CalledProcessError):
            diff_params(port, param_file_nameB)
        load_params(port, param_file_nameB)
        diff_params(port, param_file_nameB)


def test_buzzer(adrupilot_simulator):
    port = adrupilot_simulator.command
    buzzer(port, "gps-time")
    buzzer(port, "check-diff")
    buzzer(port, "git")
    buzzer(port, "planner")
    buzzer(port, "ready")


def mavlink_radio_collection_base_command(collect_port):
    # --drone-uri overrides tests/rover_config.yaml's `drone-uri:
    # tcp:127.0.0.1:14591`. Without it the collector dials a hardcoded 14591 --
    # which, once the sim moved to an ephemeral port, would be the dev SITL on
    # this box rather than the sim under test.
    return (
        f"python3 {mavlink_radio_collection.__file__} -c {root_dir}/tests/rover_config.yaml "
        f"-m {root_dir}/tests/device_mapping "
        f"--drone-uri tcp:127.0.0.1:{collect_port}"
    )


def test_manual_mode_stationary(adrupilot_simulator):
    set_mode("manual", adrupilot_simulator.command, sleep_time=10)
    collector = mavlink_radio_collection_base_command(adrupilot_simulator.collect)
    with tempfile.TemporaryDirectory() as tmpdirname:
        output = subprocess.check_output(
            f"{collector}  -r circle --temp {tmpdirname} -s 30",
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
    set_mode("manual", adrupilot_simulator.command, sleep_time=10)
    collector = mavlink_radio_collection_base_command(adrupilot_simulator.collect)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cmd = f"{collector}  -r circle --temp {tmpdirname}"
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
                    set_mode("guided", adrupilot_simulator.command)
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

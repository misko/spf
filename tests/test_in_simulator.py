import glob
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from pymavlink import mavutil

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

        # An undisturbed run must SAY it was undisturbed. Absent attributes are
        # indistinguishable from a capture written by an older build, so the
        # zeroes are the evidence -- and writing them here proves the whole
        # path from the drone's live state to the artifact, at no extra sim time.
        assert z.attrs["planner_control_lost_seconds"] == 0.0
        assert z.attrs["navigation_unhealthy_seconds"] == 0.0


# ---------------------------------------------- interrupting a guided run ----
#
# A capture is only usable while two things hold: the planner is driving, and
# the vehicle knows where it is. Both fail routinely in the field -- an
# operator takes MANUAL, a GPS drops out, a compass goes unhealthy -- and on
# 2026-08-07 the collector noticed none of it and recorded straight through.
#
# These tests interrupt a real ArduPilot mid-GUIDED, then put it back, and
# require the collector to both notice and RESUME. Recovery is half the point:
# a capture that stops trusting itself permanently after one transient blip is
# no more usable than one that never noticed.

SIM_PARAM_TIMEOUT = 30


def sim_param_set(port, name, value):
    """Set one SITL parameter, and confirm the vehicle echoed the new value.

    param_set_send rather than the controller's --load-params: loading a whole
    parameter file against this sim takes ~25s, which is longer than the
    interruptions these tests are trying to time.

    The connection is opened and closed around each call on purpose. Each sim
    endpoint is a single-client `tcpin` server, so holding one open here would
    lock out set_mode()'s subprocess and the failure would look like a hang.
    """
    connection = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}")
    try:
        connection.wait_heartbeat(timeout=SIM_PARAM_TIMEOUT)
        connection.mav.param_set_send(
            connection.target_system,
            connection.target_component,
            name.encode(),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        deadline = time.time() + SIM_PARAM_TIMEOUT
        while time.time() < deadline:
            message = connection.recv_match(
                type="PARAM_VALUE", blocking=True, timeout=5
            )
            if message is not None and message.param_id == name:
                return message.param_value
        raise TimeoutError(f"{name} was never echoed back after being set to {value}")
    finally:
        connection.close()


@dataclass(frozen=True)
class Interruption:
    """One way a guided run gets interrupted, and how it is expected to read.

    `lost_marker`/`resumed_marker` are the collector's own log lines, because
    those are what an operator standing in a field actually sees.
    """

    name: str
    inject: Callable[[int], None]
    clear: Callable[[int], None]
    lost_marker: str
    resumed_marker: str


INTERRUPTIONS = [
    # The rover 4 case: the operator flips CH8 and drives it himself. A
    # designed interlock, not a fault -- but the capture must stop counting
    # those records as planner-driven.
    Interruption(
        name="operator_takes_manual",
        inject=lambda port: set_mode("manual", port),
        clear=lambda port: set_mode("guided", port),
        lost_marker="PLANNER CONTROL LOST",
        resumed_marker="PLANNER CONTROL RESUMED",
    ),
    # gps_lat/gps_long are the ground truth every record is labelled with, so
    # recording through a GPS dropout produces confidently mislabelled data.
    #
    # Observed here: ArduPilot drops the rover into HOLD on the EKF failsafe,
    # so this trips PLANNER CONTROL LOST as well -- correctly, since nobody is
    # driving either. Only the navigation markers are asserted because the
    # control half does NOT come back on its own: restoring GPS leaves the
    # vehicle in HOLD, and the planner only recovers via the stall watchdog's
    # handover, which waits for an operator. That is by design -- an EKF
    # failsafe should not silently resume itself -- but it does mean a GPS
    # dropout in the field ends the capture's useful portion until someone
    # intervenes.
    Interruption(
        name="gps_loss",
        inject=lambda port: sim_param_set(port, "SIM_GPS_DISABLE", 1),
        clear=lambda port: sim_param_set(port, "SIM_GPS_DISABLE", 0),
        lost_marker="NAVIGATION UNHEALTHY",
        resumed_marker="NAVIGATION RECOVERED",
    ),
    # The rover 4 boot fault, now mid-run: heading feeds the ground truth too.
    Interruption(
        name="compass_loss",
        inject=lambda port: sim_param_set(port, "SIM_MAG1_FAIL", 1),
        clear=lambda port: sim_param_set(port, "SIM_MAG1_FAIL", 0),
        lost_marker="NAVIGATION UNHEALTHY",
        resumed_marker="NAVIGATION RECOVERED",
    ),
]


def _run_until_interrupted_and_resumed(endpoints, interruption, backstop_seconds=300):
    """Drive a real capture through inject -> notice -> clear -> resume.

    Returns the collector's output lines. The capture is stopped as soon as the
    resume is seen rather than run to completion: what is under test is the
    transition, and every extra second here is a second of CI.
    """
    set_mode("manual", endpoints.command, sleep_time=10)
    collector = mavlink_radio_collection_base_command(endpoints.collect)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # No shell: terminate() has to reach python itself, not an intervening
        # /bin/sh that would leave the capture running and the test hanging.
        command = shlex.split(
            f"{collector} -r circle --temp {tmpdirname} -s {backstop_seconds}"
        )
        outputs = []
        stage = "waiting_for_guided"
        with subprocess.Popen(
            command,
            env=get_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            try:
                for line in process.stdout:
                    outputs.append(line)
                    print(line, end="")

                    if (
                        stage == "waiting_for_guided"
                        and "waiting for rover to move into guided mode..." in line
                    ):
                        set_mode("guided", endpoints.command)
                        stage = "guided"
                    elif (
                        stage == "guided"
                        and "MavRadioCollection: Planner has started controling" in line
                    ):
                        # Interrupt only once recording is actually under way;
                        # before that there is nothing to record through.
                        print(f"INJECTING {interruption.name}")
                        interruption.inject(endpoints.command)
                        stage = "injected"
                    elif stage == "injected" and interruption.lost_marker in line:
                        print(f"CLEARING {interruption.name}")
                        interruption.clear(endpoints.command)
                        stage = "cleared"
                    elif stage == "cleared" and interruption.resumed_marker in line:
                        stage = "resumed"
                        break
            finally:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)

    return stage, outputs


@pytest.mark.parametrize(
    "interruption", INTERRUPTIONS, ids=lambda i: i.name
)
def test_a_guided_run_notices_an_interruption_and_resumes(
    adrupilot_simulator, interruption
):
    stage, outputs = _run_until_interrupted_and_resumed(
        adrupilot_simulator, interruption
    )
    transcript = "".join(outputs)

    assert stage != "waiting_for_guided", "the collector never asked for guided mode"
    assert stage != "guided", (
        "recording never started, so the interruption was never injected"
    )
    assert interruption.lost_marker in transcript, (
        f"{interruption.name} was injected and the capture recorded straight "
        f"through it -- no {interruption.lost_marker!r} in the log"
    )
    assert interruption.resumed_marker in transcript, (
        f"{interruption.name} was cleared but the capture never recovered -- no "
        f"{interruption.resumed_marker!r} in the log. A capture that never "
        "trusts itself again after one blip is as unusable as one that never "
        "noticed."
    )
    assert stage == "resumed"


def test_the_operator_can_hand_control_back_and_forth_repeatedly(adrupilot_simulator):
    """One takeover is the easy case; the flag must not latch after the first.

    `planner_in_control` used to be a latch set once by run_planner. Reading
    the mode live means every handback works, not just the first -- and the
    screenshots from 2026-08-07 show the switch being worked repeatedly.
    """
    set_mode("manual", adrupilot_simulator.command, sleep_time=10)
    collector = mavlink_radio_collection_base_command(adrupilot_simulator.collect)

    with tempfile.TemporaryDirectory() as tmpdirname:
        command = shlex.split(f"{collector} -r circle --temp {tmpdirname} -s 300")
        outputs = []
        stage = "waiting_for_guided"
        cycles_wanted = 2
        losses = 0
        resumes = 0
        with subprocess.Popen(
            command,
            env=get_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            try:
                for line in process.stdout:
                    outputs.append(line)
                    print(line, end="")
                    if (
                        stage == "waiting_for_guided"
                        and "waiting for rover to move into guided mode..." in line
                    ):
                        set_mode("guided", adrupilot_simulator.command)
                        stage = "driving"
                    elif (
                        stage == "driving"
                        and "MavRadioCollection: Planner has started controling" in line
                    ):
                        set_mode("manual", adrupilot_simulator.command)
                        stage = "took_manual"
                    elif stage == "took_manual" and "PLANNER CONTROL LOST" in line:
                        losses += 1
                        set_mode("guided", adrupilot_simulator.command)
                        stage = "gave_back"
                    elif stage == "gave_back" and "PLANNER CONTROL RESUMED" in line:
                        resumes += 1
                        if resumes >= cycles_wanted:
                            break
                        set_mode("manual", adrupilot_simulator.command)
                        stage = "took_manual"
            finally:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)

    assert losses >= cycles_wanted, (
        f"only {losses} of {cycles_wanted} takeovers were noticed; the control "
        "signal latched after the first"
    )
    assert resumes >= cycles_wanted, (
        f"only {resumes} of {cycles_wanted} handbacks resumed the capture"
    )

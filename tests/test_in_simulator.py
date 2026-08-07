import glob
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from pymavlink import mavutil

import docker
import spf.mavlink.mavlink_controller
from spf.mavlink import mavlink_controller
from spf import mavlink_radio_collection
from spf.dataset.v4_data import v4rx_f64_keys
from spf.mavlink.mavlink_controller import (
    Drone,
    connect_with_heartbeat,
    mavlink_connection_factory,
    meters_to_degrees,
)
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


# ------------------------------------------ commands the vehicle threw away ---
#
# DO_REPOSITION carries a flags word in param2, and spf sent 0. ArduPilot's
# Rover/GCS_Mavlink.cpp handle_command_int_do_reposition() opens with
#
#   const bool change_modes = ((int32_t)packet.param2 &
#         MAV_DO_REPOSITION_FLAGS_CHANGE_MODE) == MAV_DO_REPOSITION_FLAGS_CHANGE_MODE;
#   if (!rover.control_mode->in_guided_mode() && !change_modes) {
#       return MAV_RESULT_DENIED;
#   }
#
# so every waypoint issued while the vehicle was in anything but GUIDED was
# refused outright. HOLD is not a hypothetical there: the EKF failsafe parks the
# rover in HOLD (see the gps_loss interruption above), _escape_jam passes
# through HOLD by design, and a capture abort sends HOLD itself.
#
# spf could not have noticed either way. handle_COMMAND_ACK was a bare `pass`,
# so "obeyed" and "refused" were the same event -- silence -- for every command
# this controller has ever sent, arm() included. The 939 s parked run on
# 2026-08-07 was a different mechanism (SubMode::Stop, see move_to_point), but
# it was invisible for this same reason, and it stays invisible until somebody
# reads the acks.
#
# Both halves are pinned here: the command has to be accepted and the wheels
# have to turn, and a genuine refusal has to be loud.

# At -S 5 with WP_SPEED 1.5 the rover covers ~7.5 m per wall-clock second, so a
# leg this long is still being driven for the whole window measured below. The
# number is sized from a failure: a 42 m target was reached before the phase
# under measurement, and the test then "passed" on a parked vehicle.
REPOSITION_LEG_M = 200

# Far enough past GPS noise and past the ~5 m WP_RADIUS that "it moved" cannot
# be a stationary rover's position estimate wandering.
REPOSITION_MOVED_M = 10


def wait_until(predicate, timeout, poll=0.1):
    """True as soon as `predicate` holds; False if it never does in `timeout`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return bool(predicate())


def drone_set_mode(drone, mode, timeout=15):
    """Set a mode and wait for the heartbeat to confirm it.

    Retries the way set_drone_mode() in the controller does: a single SET_MODE
    can be dropped, and a mode that never took would otherwise surface as a
    baffling assertion three steps later.
    """
    expected = f"ROVER_MODE_{mode}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        drone.set_mode(mode)
        if wait_until(lambda: drone.mav_mode == expected, 2.0):
            return True
    return drone.mav_mode == expected


def drone_arm(drone, timeout=20):
    """Arm and wait for the heartbeat's ARMED flag, retrying the command."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        drone.arm()
        if wait_until(lambda: drone.armed, 2.0):
            return True
    return bool(drone.armed)


@contextmanager
def commanding_drone(port):
    """A real Drone bound to the sim's command endpoint, torn down cleanly.

    In-process rather than through a `mavlink_controller.py` subprocess, unlike
    set_mode() above, for two reasons: the CLI has no reposition, and what is
    under test is what the Drone RECORDS about the vehicle's answer -- a
    subprocess would only ever show an exit status.

    `connection_factory` is deliberately NOT passed. With one, the receive
    loop's reconnect would re-dial the endpoint the moment this closes it, and
    each `tcpin` accepts a single client: a leftover reconnecting Drone would
    hold the command port and the next test would die with "no heartbeat",
    which reads as a simulator fault and is not one. Without one,
    _recover_connection returns False and the loop thread exits on the close.
    """
    endpoint = f"tcp:127.0.0.1:{port}"
    connection, heartbeat = connect_with_heartbeat(mavlink_connection_factory(endpoint))
    drone = Drone(connection)
    drone.process_message(heartbeat)
    drone.start()
    try:
        # The production readiness gate (fix, healthy GPS sensor, converged
        # EKF), waited on for a reason that cost a confusing run: a rover that
        # is ARMED can only enter GUIDED if ArduPilot's Mode::enter() likes the
        # position estimate, so commanding a cold simulator gets
        # MAV_RESULT_FAILED out of the very handler under test -- a different
        # refusal that looks exactly like the DENIED these tests are about.
        assert wait_until(lambda: drone.drone_ready, 120), (
            "vehicle never became ready: "
            f"gps={drone.gps} sats={drone.gps_satellites} "
            f"fix={drone.gps_fix_type} ekf={drone.ekf_healthy} "
            f"health={drone.sensors_health}"
        )
        yield drone
    finally:
        # Hand the vehicle back parked and disarmed. These tests deliberately
        # send it 200 m down-range and the simulator is session-scoped, so
        # leaving it driving would become the next test's mystery.
        try:
            drone_set_mode(drone, "HOLD", timeout=6)
            drone.disarm()
        except Exception as error:  # teardown must not mask a real failure
            print(f"could not park the vehicle after the test: {error}")
        connection.close()


def reposition_errors(caplog):
    return [
        record.getMessage()
        for record in list(caplog.records)
        if "MAV_CMD_DO_REPOSITION" in record.getMessage()
    ]


def test_a_reposition_out_of_hold_is_accepted_and_drives(adrupilot_simulator, caplog):
    """A waypoint issued from HOLD has to reach the wheels, not the bit bucket.

    Defect F3 as the field would meet it: the rover sits in HOLD (EKF failsafe,
    escape maneuver, operator), the planner issues its next waypoint exactly as
    it always does, and with param2=0 the autopilot answers MAV_RESULT_DENIED
    and the rover never moves.

    The second half re-issues from GUIDED, because a flag that fixes HOLD by
    breaking the path that already worked would be a worse bug than the one it
    fixes.
    """
    with commanding_drone(adrupilot_simulator.command) as drone, caplog.at_level(
        logging.ERROR
    ):
        assert drone_set_mode(
            drone, "HOLD"
        ), f"vehicle never entered HOLD; mode is {drone.mav_mode}"
        # Armed in HOLD on purpose. HOLD holds the vehicle still whatever the
        # throttle says, so arming first leaves the command itself as the only
        # thing standing between the test and motion.
        assert drone_arm(drone), "vehicle never armed in HOLD, so nothing can move"

        start = np.array(drone.gps, dtype=float)  # (long, lat)
        target = start + meters_to_degrees(0, REPOSITION_LEG_M, start[1])
        drone.reposition(lat=target[1], long=target[0])

        entered_guided = wait_until(lambda: drone.mav_mode == "ROVER_MODE_GUIDED", 15)
        drove = wait_until(
            lambda: drone.distance_to_target(start) > REPOSITION_MOVED_M, 20
        )
        travelled = drone.distance_to_target(start)

        assert entered_guided, (
            "the autopilot refused the waypoint and stayed in "
            f"{drone.mav_mode}: DO_REPOSITION went out with param2=0, so "
            "Rover/GCS_Mavlink.cpp returned MAV_RESULT_DENIED without ever "
            "looking at the destination"
        )
        assert drove, (
            f"vehicle moved only {travelled:.1f}m of the {REPOSITION_MOVED_M}m "
            "that would prove the waypoint reached the wheels"
        )

        # The already-in-GUIDED path, unchanged: still accepted, and it still
        # retargets rather than bouncing the mode and dropping the destination
        # the way a GUIDED re-entry does.
        here = np.array(drone.gps, dtype=float)
        second = here + meters_to_degrees(REPOSITION_LEG_M, 0, here[1])
        before = drone.distance_to_target(second)
        drone.reposition(lat=second[1], long=second[0])
        closed_in = wait_until(
            lambda: before - drone.distance_to_target(second) > REPOSITION_MOVED_M,
            25,
        )
        assert (
            drone.mav_mode == "ROVER_MODE_GUIDED"
        ), f"a reposition from GUIDED knocked the vehicle into {drone.mav_mode}"
        assert closed_in, (
            "a reposition sent from GUIDED did not retarget the vehicle; it "
            f"closed only {before - drone.distance_to_target(second):.1f}m"
        )

        assert (
            reposition_errors(caplog) == []
        ), "an accepted reposition must not be reported as a refusal"

        # The ack is now state, not only a log line, so a caller can act on it.
        # Asserted last: the behaviour above is the claim, this is how spf comes
        # to know it.
        recorded = getattr(drone, "command_results", {}).get("MAV_CMD_DO_REPOSITION")
        assert recorded is not None, "no COMMAND_ACK was recorded for the reposition"
        assert (
            recorded.result == "MAV_RESULT_ACCEPTED"
        ), f"the vehicle answered {recorded.result} to the reposition"


def test_a_refused_command_is_not_silent(adrupilot_simulator, caplog):
    """A refusal spf cannot see is a refusal spf can never retry.

    (0, 0) is refused by the SAME handler under test, a few lines below the
    mode check that F3 is about:

        if (packet.x == 0 && packet.y == 0) { return MAV_RESULT_DENIED; }

    so it stays a genuine autopilot refusal once the CHANGE_MODE flag is set --
    this test cannot be made to pass by that flag, only by reading the acks --
    and it moves the vehicle nowhere, which is why it is safe to send.
    """
    with commanding_drone(adrupilot_simulator.command) as drone, caplog.at_level(
        logging.ERROR
    ):
        assert drone_set_mode(
            drone, "HOLD"
        ), f"vehicle never entered HOLD; mode is {drone.mav_mode}"
        caplog.clear()

        drone.reposition(lat=0.0, long=0.0)

        logged = wait_until(
            lambda: any(
                "MAV_CMD_DO_REPOSITION" in message and "DENIED" in message
                for message in reposition_errors(caplog)
            ),
            10,
        )
        assert logged, (
            "the autopilot refused the reposition with MAV_RESULT_DENIED and "
            "spf logged nothing: handle_COMMAND_ACK is a bare `pass`, so every "
            "command this controller sends -- arm() included -- is fire and "
            f"forget. ERRORs seen: {[r.getMessage() for r in caplog.records]}"
        )

        recorded = getattr(drone, "command_results", {}).get("MAV_CMD_DO_REPOSITION")
        assert (
            recorded is not None
        ), "nothing in the Drone's state records that the command was refused"
        assert (
            recorded.result == "MAV_RESULT_DENIED"
        ), f"expected the refusal to be recorded, got {recorded.result}"


def test_a_reposition_never_takes_the_vehicle_out_of_the_operators_hands(
    adrupilot_simulator, caplog
):
    """MANUAL is the one mode the mode-change flag must NOT override.

    The flag that fixes HOLD would, applied unconditionally, let the planner's
    next waypoint yank the rover out of an operator's hands mid-takeover -- and
    ArduPilot's MODE_CH only re-applies on switch MOVEMENT, so the operator
    would keep holding a CH8 that says MANUAL while the vehicle drove itself.
    That is the interlock _hand_over_to_manual is built on and the switch the
    rover-4 operator worked repeatedly on 2026-08-07.

    So this pins the refusal as DESIRED behaviour: from MANUAL the command goes
    out with no flag, the autopilot denies it, the vehicle stays exactly where
    the human left it -- and, unlike before, spf says so out loud.

    It also demonstrates the defect directly on the firmware: the ONLY
    difference between this DENIED and the ACCEPTED above is param2.
    """
    with commanding_drone(adrupilot_simulator.command) as drone, caplog.at_level(
        logging.ERROR
    ):
        assert drone_set_mode(
            drone, "MANUAL"
        ), f"vehicle never entered MANUAL; mode is {drone.mav_mode}"
        caplog.clear()

        here = np.array(drone.gps, dtype=float)
        target = here + meters_to_degrees(0, REPOSITION_LEG_M, here[1])
        drone.reposition(lat=target[1], long=target[0])

        grabbed = wait_until(lambda: drone.mav_mode != "ROVER_MODE_MANUAL", 5)
        assert not grabbed, (
            "a planner waypoint pulled the vehicle out of MANUAL and into "
            f"{drone.mav_mode} while an operator was driving it"
        )
        assert any(
            "DENIED" in message for message in reposition_errors(caplog)
        ), f"the refusal was not logged; ERRORs seen: {reposition_errors(caplog)}"






# ------------------------------------- an operator excursion mid-waypoint ----
#
# Noticing a takeover is not the same as surviving one. The tests above prove
# the CAPTURE flags the excursion; this one proves the VEHICLE comes back.
#
# ArduPilot discards the GUIDED destination on every re-entry into GUIDED:
# ModeGuided::_enter() calls start_stop() (Rover/mode_guided.cpp:3-20), which
# sets SubMode::Stop (:392), and stop_vehicle() then holds both throttle
# channels at exactly 1500 (Rover/mode.cpp:336-363). start_stop() has only
# those two call sites inside _enter(), so SubMode::Stop is reachable ONLY by
# re-entering GUIDED -- which is precisely what an operator does with CH8:
# takes MANUAL, drives, hands it back. Every ArduRover >= 4.2.0 does this.
#
# move_to_point used to issue its DO_REPOSITION exactly once, before the loop
# (mavlink_controller.py:1516), and the loop had no re-issue, no deadline and
# no mode check. So the rover returned to GUIDED with no destination and sat
# there. On 2026-08-07 RO1 held ONE waypoint for 939 s after a CH8 takeover and
# wrote 1101 of 3000 records parked. Nothing in the log looked wrong, because
# from the Pi's side nothing was: armed, GUIDED, EKF healthy, motionless.
#
# The stall watchdog cannot cover it either, and that is not an oversight to be
# fixed there: `driving` (mavlink_controller.py:1262) requires motor_active,
# which is False exactly when the autopilot commands neutral throttle -- so the
# parked-in-GUIDED state resets the watchdog's own anchor on every tick and its
# clock can never run. The recovery has to live in move_to_point.
#
# Hence: drive a real leg, flip GUIDED -> MANUAL -> GUIDED with NOTHING else
# sent, and require the rover to reach its ORIGINAL target. The assertion is on
# ground actually closed, never on a log line -- a correct-looking log beside a
# motionless vehicle IS the failure mode. An earlier attempt at this test
# "passed" on 1.047 m of GPS jitter against a 1.0 m threshold; the margin here
# is ~70 m of closed distance, which no jitter can manufacture.

# The leg has to be long enough that the rover is still well short of it when
# the excursion ends -- a 42 m target was reached before the interesting phase
# in an earlier attempt. Measured here at -S 5: ~12 m/s of wall-clock ground
# speed, so 120 m is ~10 s of driving and the handback happens with ~84 m left.
EXCURSION_LEG_M = 120.0
# Take MANUAL only once the rover is demonstrably under way. Before that a
# frozen rover and a rover that never started are indistinguishable.
EXCURSION_TRIGGER_M = 25.0
# Long enough for the vehicle to actually coast to a stop in MANUAL (no RC
# input, so MANUAL means zero throttle) and for at least two 1 Hz heartbeats to
# carry the mode change, which is how move_to_point observes it at all.
EXCURSION_MANUAL_SECONDS = 3.0
# Generous against measurement, not against the defect: the remaining ~84 m
# takes ~7 s at the SITL default speed and ~11 s if rover3_base_parameters has
# been loaded by test_load_and_diff_params (WP_SPEED 1.5 -> 7.5 m/s wall). A
# stranded rover closes nothing at all, so this bound only costs time in RED.
EXCURSION_RESUME_SECONDS = 45.0
# Above rover3_base_parameters' WP_RADIUS of 5.0. Under it the vehicle stops on
# its own radius and move_to_point would spin forever on a rover that arrived --
# the same class of bug as the 3 m escape legs in test_in_simulator_crash.py.
EXCURSION_TOLERANCE_M = 10.0


class _Leg(threading.Thread):
    """`move_to_point` running where the test can watch it, as the planner does.

    The production call is blocking and, on the defect, blocks forever -- so it
    cannot be called inline. Errors are captured rather than raised: teardown
    closes the link out from under a stranded leg on purpose, and that
    exception is expected, not a finding.
    """

    def __init__(self, drone, target):
        super().__init__(daemon=True)
        self.drone = drone
        self.target = target
        self.outcome = None
        self.error = None

    def run(self):
        try:
            self.outcome = self.drone.move_to_point(self.target)
        except BaseException as error:  # noqa: BLE001 -- reported, never raised
            self.error = error


def _rover_side_drone(port):
    """The rover's own controller, attached to the sim like the collector's."""

    factory = mavlink_controller.mavlink_connection_factory(f"tcp:127.0.0.1:{port}")
    connection, heartbeat = mavlink_controller.connect_with_heartbeat(
        factory, attempts=5
    )
    drone = mavlink_controller.Drone(
        connection,
        tolerance_in_m=EXCURSION_TOLERANCE_M,
        # connection_factory is deliberately NOT handed on. With one, the
        # receive loop redials on any error -- including the close() that ends
        # this test -- and would re-claim the single-client endpoint that the
        # next test's collector needs. Without one, close() simply ends the
        # loop.
        connection_factory=None,
        # The stall watchdog is not what is under test, and as argued above it
        # provably cannot fire on a rover parked at neutral throttle. Leaving
        # it on would only add a way for the test to HANG instead of fail: a
        # STALL_MANUAL verdict parks the rover and waits for an operator who is
        # never coming.
        crash_detect=False,
    )
    drone.process_message(heartbeat)
    return drone.start()


def _wait_until(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {description}")


def _arm(drone, timeout=60):
    """Arming is a request, not a setter -- retry until the heartbeat agrees."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if drone.armed:
            return
        drone.arm()
        time.sleep(1.0)
    raise AssertionError(f"the vehicle never armed within {timeout:.0f}s")


def test_an_operator_excursion_mid_leg_does_not_strand_the_rover(adrupilot_simulator):
    # Roles, not positions: the rover-side controller takes COLLECT and the
    # "operator" commands through COMMAND. Each tcpin serves exactly one
    # client, so sharing one endpoint would hang rather than fail.
    set_mode("manual", adrupilot_simulator.command)
    drone = _rover_side_drone(adrupilot_simulator.collect)
    leg = None
    try:
        _wait_until(
            lambda: drone.gps[0] != 0 and drone.ekf_healthy,
            60,
            "a usable position from the simulator",
        )
        # Due north of wherever the previous test left the rover, so this does
        # not depend on the session's history.
        target = drone.gps + mavlink_controller.meters_to_degrees(
            0.0, EXCURSION_LEG_M, drone.gps[1]
        )

        set_mode("guided", adrupilot_simulator.command)
        _wait_until(
            lambda: drone.mav_mode == "ROVER_MODE_GUIDED", 30, "GUIDED mode"
        )
        _arm(drone)

        start_distance = drone.distance_to_target(target)
        leg = _Leg(drone, target)
        leg.start()
        _wait_until(
            lambda: start_distance - drone.distance_to_target(target)
            > EXCURSION_TRIGGER_M,
            60,
            f"the rover to cover the first {EXCURSION_TRIGGER_M:.0f}m of the leg",
        )

        # The excursion. NOTHING else is sent -- no destination, no arm, no
        # mode nudge beyond the operator's two flips. That is the whole point:
        # in the field the operator's CH8 switch is the only input.
        set_mode("manual", adrupilot_simulator.command)
        _wait_until(
            lambda: drone.mav_mode == "ROVER_MODE_MANUAL", 30, "the operator's MANUAL"
        )
        time.sleep(EXCURSION_MANUAL_SECONDS)
        set_mode("guided", adrupilot_simulator.command)
        _wait_until(
            lambda: drone.mav_mode == "ROVER_MODE_GUIDED", 30, "the handback to GUIDED"
        )

        handback_distance = drone.distance_to_target(target)
        closest_distance = handback_distance
        deadline = time.monotonic() + EXCURSION_RESUME_SECONDS
        while time.monotonic() < deadline and leg.outcome is None:
            closest_distance = min(closest_distance, drone.distance_to_target(target))
            time.sleep(0.2)
        final_distance = drone.distance_to_target(target)
        closest_distance = min(closest_distance, final_distance)
        outcome, error, armed, mode = leg.outcome, leg.error, drone.armed, drone.mav_mode
    finally:
        # Close first, so the stranded leg's next command fails fast instead of
        # holding the endpoint for the rest of the session.
        try:
            drone.connection.close()
        except Exception:
            pass
        if leg is not None:
            leg.join(timeout=15)
        set_mode("manual", adrupilot_simulator.command)
        # One tcpin, one client: give the sim a moment to notice this one left.
        time.sleep(2)

    report = (
        f"leg={EXCURSION_LEG_M:.1f}m start={start_distance:.2f}m "
        f"handback={handback_distance:.2f}m closest={closest_distance:.2f}m "
        f"final={final_distance:.2f}m "
        f"closed_after_handback={handback_distance - closest_distance:.2f}m "
        f"outcome={outcome} armed={armed} mode={mode} leg_error={error!r}"
    )
    assert closest_distance <= EXCURSION_TOLERANCE_M, (
        "the rover never got back to its original target after a MANUAL "
        "excursion -- ArduPilot dropped the destination on re-entering GUIDED "
        f"and move_to_point never re-issued it. {report}"
    )
    assert outcome == mavlink_controller.MOVE_REACHED, (
        "the rover closed on the target but move_to_point did not report "
        f"reaching it. {report}"
    )


# ------------------------- the watchdog as a backstop, not a duplicate -------
#
# `move_to_point` now re-issues the destination whenever it observes a
# re-entry into GUIDED, which closes the 2026-08-07 mechanism at source. That
# fix and this watchdog are NOT redundant, and the difference decides how this
# test has to be written.
#
# The re-issue is edge-triggered on a mode transition, so it only ever heals a
# destination lost TO a mode transition. Anything else that leaves the rover
# parked with a waypoint outstanding -- a DO_REPOSITION denied at the far end
# (Rover/GCS_Mavlink.cpp:743 refuses one that arrives outside GUIDED), a command
# lost on the radio link, a GCS on the same link overriding the destination --
# is invisible to it. The watchdog is what covers those, and it could not: its
# gate required `motor_active`, which is False precisely when the autopilot is
# commanding neutral throttle.
#
# So this test must park the rover WITHOUT a mode transition, or the re-issue
# would heal it in one 0.1 s poll and the watchdog would never be exercised.
# It overrides the destination with the rover's OWN CURRENT POSITION: ArduPilot
# judges arrival on WP_RADIUS, so a target already inside that radius is
# "reached" the instant it is issued and the vehicle stops -- the same
# behaviour docs/learnings.md records as making a short escape leg a no-op,
# used here on purpose. The collector's `move_to_point` still has its real,
# distant waypoint outstanding and no mode ever changed.

def _terminate_quietly(process):
    """Stop the collector, escalating only if it ignores SIGTERM."""
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


PARKED_STALL_DEADLINE_SECONDS = 90.0
STALL_MARKER = "STALL: no progress for"

# Park the rover only while it is still a long way from its waypoint. The
# override stops the vehicle where it stands, but `move_to_point` only keeps
# waiting -- and therefore only keeps the destination outstanding -- while the
# target is out of reach. Park it near one and it arrives, the planner issues
# the next leg with a fresh reposition of its own, and the rover drives away
# again: measured, on the first attempt, as motor_active True throughout and
# distances cycling 5-78 m. A bounce leg at 5x sim speed is only seconds long,
# so the window to park is narrow and has to be chosen, not taken.
PARK_MIN_DISTANCE_M = 40.0
DIST_SAMPLE = re.compile(r"Dist \(m\) to target ([0-9.]+) (True|False) ([A-Z_]+)")


def test_a_rover_parked_without_a_mode_change_is_caught_by_the_stall_watchdog(
    adrupilot_simulator,
):
    """The rover stops with a waypoint outstanding and no mode transition.

    Every signal the collector watches says healthy -- armed, GUIDED, EKF
    good, planner in control -- while the vehicle sits. Before the gate was
    fixed this produced no stall line at all, which is exactly what rover 1
    did for 939 s on 2026-08-07.

    The commanding link is opened BEFORE the rover is parked and held open to
    the end. Closing it costs a MAVLink heartbeat stream, and FS_GCS_ENABLE
    then runs FS_ACTION: the first version of this test dropped the link right
    after parking and ArduPilot answered with a failsafe -- "Throttle
    disarmed", then ROVER_MODE_HOLD. That is a real behaviour worth knowing,
    but it is not this defect, and it takes the vehicle out of GUIDED so the
    gate under test never even applies.
    """
    set_mode("manual", adrupilot_simulator.command, sleep_time=10)
    collector = mavlink_radio_collection_base_command(adrupilot_simulator.collect)

    with tempfile.TemporaryDirectory() as tmpdirname:
        command = shlex.split(f"{collector} -r bounce --temp {tmpdirname} -s 400")
        outputs = []
        stage = "waiting_for_guided"
        parked_at = None
        stall_seen_while_driving = False
        with ExitStack() as stack:
            process = stack.enter_context(
                subprocess.Popen(
                    command,
                    env=get_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
            stack.callback(_terminate_quietly, process)
            for line in process.stdout:
                outputs.append(line)
                print(line, end="")

                if (
                    stage == "waiting_for_guided"
                    and "waiting for rover to move into guided mode..." in line
                ):
                    set_mode("guided", adrupilot_simulator.command)
                    stage = "guided"
                elif (
                    stage == "guided"
                    and "MavRadioCollection: Planner has started controling" in line
                ):
                    stage = "recording"
                elif stage == "recording":
                    # A stall reported while the rover is genuinely driving is
                    # a false positive, and the whole risk of taking
                    # motor_active out of the gate. Watch for it explicitly:
                    # asserting only that a stall EVENTUALLY appears would be
                    # satisfied by a watchdog that fires constantly.
                    if STALL_MARKER in line:
                        stall_seen_while_driving = True
                    sample = DIST_SAMPLE.search(line)
                    if (
                        sample is not None
                        and sample.group(2) == "True"
                        and sample.group(3) == "ROVER_MODE_GUIDED"
                        and float(sample.group(1)) > PARK_MIN_DISTANCE_M
                    ):
                        # Under way (motor_active True) AND far from the
                        # waypoint. Parking a rover that never moved proves
                        # nothing -- "no progress" would be indistinguishable
                        # from "not yet asked to move" -- and parking one that
                        # is nearly there just lets it arrive.
                        drone = stack.enter_context(
                            commanding_drone(adrupilot_simulator.command)
                        )
                        assert wait_until(
                            lambda: drone.gps is not None and drone.gps[0] != 0, 20
                        ), "no GPS on the commanding link"
                        here = drone.gps.copy()  # (long, lat)
                        drone.reposition(lat=here[1], long=here[0])
                        parked_at = time.time()
                        stage = "parked"
                elif stage == "parked":
                    if STALL_MARKER in line:
                        stage = "stalled"
                        break
                    if time.time() - parked_at > PARKED_STALL_DEADLINE_SECONDS:
                        break

    assert stage != "waiting_for_guided", "the collector never asked for guided mode"
    assert stage != "guided", "recording never started"
    assert parked_at is not None, "the rover never drove, so it was never parked"
    assert not stall_seen_while_driving, (
        "the watchdog fired while the rover was still driving -- a false stall "
        "sends an operator across a field after a healthy rover"
    )
    modes = [
        line.split()[-1]
        for line in outputs
        if "Dist (m) to target" in line and line.split()[-1].startswith("ROVER_MODE_")
    ]
    assert stage == "stalled", (
        "the rover sat parked with a waypoint outstanding for "
        f"{PARKED_STALL_DEADLINE_SECONDS:.0f}s and the watchdog never fired. "
        f"Modes seen after parking: {sorted(set(modes[-20:]))}. If those are "
        "not GUIDED the vehicle left the mode under test -- check for a "
        "failsafe -- otherwise the gate is still blind: it required "
        "motor_active, which is False exactly when the autopilot commands "
        "neutral throttle, so the failure suppresses its own detector."
    )

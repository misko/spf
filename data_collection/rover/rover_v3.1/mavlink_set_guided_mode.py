# Import mavutil
import argparse
import logging
import time

from pymavlink import mavutil

from spf.mavlink.mavlink_controller import Drone

logging.basicConfig(
    filename="logs.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s: %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--ports", nargs="+", type=int, required=True)
    parser.add_argument("-m", "--mode", type=str, required=False, default="MANUAL")

    args = parser.parse_args()
    logging.info("WTF33")
    logging.info("Connecting...")
    print(args.ports)
    drones = [
        Drone(connection=mavutil.mavlink_connection(f"udpin:127.0.0.1:{port}")).start()
        for port in args.ports
    ]
    time.sleep(3)

    for drone in drones:
        drone.set_mode(args.mode)
    for drone in drones:
        drone.set_mode(args.mode)
    for drone in drones:
        drone.set_mode(args.mode)

    time.sleep(2)

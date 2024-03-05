import argparse
import sys
import threading
import time

import RPi.GPIO as GPIO


class UtrasonicDistanceController:
    def __init__(self, trigger=2, echo=3):
        # assign pins
        self.trigger = trigger
        self.echo = echo
        self.distance = 10000

        # configure board and pins
        GPIO.setmode(GPIO.BCM)  # GPIO Mode (BOARD / BCM)
        GPIO.setup(self.trigger, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)

    def run(self, sample_interval=0.05):
        # start sensor
        GPIO.output(self.trigger, GPIO.LOW)
        time.sleep(2)

        while True:
            # send pulse
            GPIO.output(self.trigger, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(self.trigger, GPIO.LOW)

            # receive response and calculate distance
            if GPIO.wait_for_edge(self.echo, GPIO.RISING, timeout=500):
                StartTime = time.time()
                if GPIO.wait_for_edge(self.echo, GPIO.FALLING, timeout=700):
                    StopTime = time.time()
                    TimeElapsed = StopTime - StartTime
                    distance = (TimeElapsed * 34300) / 2
                    self.distance = distance
            time.sleep(sample_interval)

        # GPIO.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trigger", type=str, help="trigger", required=False, default=2
    )
    parser.add_argument("--echo", type=int, help="echo", required=False, default=3)

    args = parser.parse_args()

    udc = UtrasonicDistanceController(trigger=args.trigger, echo=args.echo)
    threading.Thread(target=udc.run, daemon=True).start()
    for x in range(15 * 2):
        time.sleep(1.0 / 2)
        print(udc.distance)
    sys.exit(1)

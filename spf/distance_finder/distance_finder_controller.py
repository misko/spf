import argparse
import sys
import threading
import time

try:
    import RPi.GPIO as GPIO
except (RuntimeError, ImportError):
    pass


class DistanceFinderController:
    def __init__(self, trigger=2, echo=3):
        # assign pins
        self.trigger = trigger
        self.echo = echo
        self.distance = 10000

        # configure board and pins
        GPIO.setmode(GPIO.BCM)  # GPIO Mode (BOARD / BCM)
        GPIO.setup(self.trigger, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)

    def run_in_new_thread(self):
        threading.Thread(target=self.run, daemon=True).start()

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

    udc = DistanceFinderController(trigger=args.trigger, echo=args.echo)
    udc.run_in_new_thread()
    for x in range(15 * 2):
        time.sleep(1.0 / 2)
        print(udc.distance)
    sys.exit(1)

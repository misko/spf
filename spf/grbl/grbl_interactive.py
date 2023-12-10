import serial
import time
import sys
import numpy as np
import matplotlib.pyplot as plt


home_pA=np.array([3568,0])
home_pB=np.array([0,0])

class GRBLManager:
    # a is the left motor on the wall
    # b is the right motor on the wall
    # (a-b)_x > 0
    # x is positive from b towards a
    # y is positive up to ceiling and negative down to floor

    def from_steps(self, a_motor_steps, b_motor_steps):
        r1 = np.linalg.norm(self.pA) + a_motor_steps
        r2 = np.linalg.norm(self.pB) + b_motor_steps
        d = np.linalg.norm(self.pA - self.pB)

        '''
        x^2 = y^2 + r_2^2
        (x-d)^2 = x^2 - 2xd + d^2 = y^2 + r_1^2

        x = (r_2^2 - d^2 - r_1^2 ) / (2*d)

        '''
        x = (r2 * r2 - d * d - r1 * r1) / (2 * d)  # on different axis
        x_dir = (self.pA - self.pB) / d # unit vector from pB to pA

        y = np.sqrt(r1 * r1 - x * x)
        y_dir = np.array([-x_dir[1], x_dir[0]])  # x_dir.T # orthogonal to x

        return self.pA + y_dir * y + x_dir * x

    def to_steps(self, p):
        a_motor_steps = np.linalg.norm(self.pA-p) - np.linalg.norm(self.pA)
        b_motor_steps = np.linalg.norm(self.pB-p) - np.linalg.norm(self.pB)

        return a_motor_steps, b_motor_steps

    def spiral(self):
        center = np.array([1500, 900])
        spiral_radius = 1200
        t_max = 6 * 2 * np.pi
        v = spiral_radius / t_max
        w = 1
        for t in np.linspace(0, t_max, 256 * 16 * 2):
            x = (v * t) * np.cos(w * t)
            y = (v * t) * np.sin(w * t)
            p = np.array([x, y]) + center
            a_motor_steps, b_motor_steps = self.to_steps(p)
            cmd = "G0 X%0.2f Y%0.2f" % (b_motor_steps, a_motor_steps)
            # print("SENDING",x,y,cmd)
            self.s.write((cmd + "\n").encode())  # Send g-code block to grbl
            self.s.readline().strip()

    def calibrate(self):
        for x in np.linspace(1500, 500, 5):
            for y in np.linspace(30, 1200, 5):
                x, y = self.to_steps(np.array([x, y]))
                cmd = "G0 X%0.2f Y%0.2f" % (x, y)
                self.s.write((cmd + "\n").encode())  # Send g-code block to grbl

    def __init__(self, serial_fn, pA, pB,bounding_box,motor_mapping):
        self.pA=pA
        self.pB=pB
        self.bounding_box=bounding_box
        self.motor_mapping=motor_mapping
        if serial_fn is not None:
          # Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
          self.s = serial.Serial(
              serial_fn, 115200, timeout=0.3, write_timeout=0.3
          )  # GRBL operates at 115200 baud. Leave that part alone.
          self.s.write("?".encode())
          grbl_out = self.s.readline()  # get the response
          print("GRBL ONLINE", grbl_out)
          self.position = {"time": time.time(), "xy": np.zeros(2)}
          self.update_status()
          time.sleep(0.05)

    def push_reset(self):
        self.s.write(b"\x18")
        return self.s.readline().decode().strip()

    def update_status(self, skip_write=False):
        if not skip_write:
            time.sleep(0.01)
            self.s.write("?".encode())
        time.sleep(0.01)

        start_time = time.time()
        response = self.s.readline().decode().strip()
        time.sleep(0.01)
        # print("STATUS",response)
        # <Idle|MPos:-3589.880,79.560,0.000,0.000|FS:0,0>
        try:
            motor_position_str = response.split("|")[1]
        except Exception as e:
            print("FAILED TO PARSE", response, "|e|", e, time.time() - start_time)
            return self.update_status(skip_write=not skip_write)
        b_motor_steps, a_motor_steps, _, _ = map(
            float, motor_position_str[len("MPos:") :].split(",")
        )

        xy = self.from_steps(a_motor_steps, b_motor_steps)
        is_moving = (self.position["xy"] != xy).any()
        self.position = {
            "a_motor_steps": a_motor_steps,
            "b_motor_steps": b_motor_steps,
            "xy": xy,
            "is_moving": is_moving,
            "time": time.time(),
        }
        return self.position

    def wait_while_moving(self):
        while True:
            old_pos = self.update_status()
            time.sleep(0.05)
            new_pos = self.update_status()
            if (
                old_pos["a_motor_steps"] == new_pos["a_motor_steps"]
                and old_pos["b_motor_steps"] == new_pos["b_motor_steps"]
            ):
                return
            time.sleep(0.01)

    def binary_search_edge(self, left, right, xy, direction, epsilon):
        if (right - left) < epsilon:
            return left
        l = (right + left) / 2
        p = l * direction + xy
        steps = self.to_steps(np.copy(p))
        actual = self.from_steps(*steps)
        deviation = np.linalg.norm(p - actual)
        if deviation > 0.0001:
            # go back left
            return self.binary_search_edge(left, l, xy, direction, epsilon)
        return self.binary_search_edge(l, right, xy, direction, epsilon)

    def bounce(self, bounces, direction=None):
        if direction is None:
            theta = np.random.uniform(2 * np.pi)
            direction = np.array([np.sin(theta), np.cos(theta)])
        percent_random = 0.05
        theta = np.random.uniform(2 * np.pi)
        direction = (1 - percent_random) * direction + percent_random * np.array(
            [np.sin(theta), np.cos(theta)]
        )
        direction = direction / np.linalg.norm(direction)
        print("bounce", direction, np.linalg.norm(direction))
        for _ in range(bounces):
            to_points, new_direction = self.single_bounce(direction)
            if len(to_points) > 0:
                print("MOVE", to_points[0], to_points[-1])
            for point in to_points:
                self.move_to(point)
                # print("MOVE")
                self.update_status()
                while np.linalg.norm(self.position["xy"] - point) > 100:
                    self.update_status()
            if (new_direction != direction).any():  # we are changing direction
                self.wait_while_moving()
                direction = new_direction
        return direction

    def single_bounce(self, direction, xy=None, step_size=5):
        # find current position
        # pick a random direction
        # take full field step
        # if hit a wall
        # direction=np.array([1,0])
        # direction=direction/np.linalg.norm(direction)
        if xy is None:
            self.update_status()
            xy = self.position["xy"]
        # find out at what point xy+l*direction we stop changing one of the variables
        epsilon = 0.001
        l = self.binary_search_edge(0, 10000, xy, direction, epsilon)
        # find a paralell vector to the boundary
        p1 = self.from_steps(*self.to_steps((l + 2 * epsilon) * direction + xy))
        p2 = self.from_steps(*self.to_steps((l + 3 * epsilon) * direction + xy))
        if np.linalg.norm(p1 - p2) < epsilon**2:  # the direction is only X or Y
            new_direction = -direction
        else:
            b = p2 - p1
            b /= np.linalg.norm(b)
            bn = np.array([-b[1], b[0]])
            _xy = self.from_steps(*self.to_steps(xy))
            if (
                np.linalg.norm(self.from_steps(*self.to_steps(_xy + bn)) - _xy)
                < epsilon
            ):
                bn = -bn
            new_direction = np.dot(direction, b) * b - np.dot(direction, bn) * bn
        to_points = []
        _l = 0
        while _l < l:
            _l = min(_l + step_size, l)
            to_points.append(_l * direction + xy)
        theta = np.random.uniform(2 * np.pi)
        percent_random = 0.05
        new_direction = (
            1 - percent_random
        ) * new_direction + percent_random * np.array([np.sin(theta), np.cos(theta)])
        return to_points, new_direction

    def move_to(self, p):
        a_motor_steps, b_motor_steps = self.to_steps(p)
        cmd = "G0 X%0.2f Y%0.2f" % (b_motor_steps, a_motor_steps)
        time.sleep(0.01)
        self.s.write((cmd + "\n").encode())  # Send g-code block to grbl
        time.sleep(0.01)
        grbl_out = self.s.readline()  # Wait for grbl response with carriage return
        time.sleep(0.01)
        # print("MOVE TO RESPONSE", grbl_out)

    def close(self):
        self.s.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("grblman: %s device" % sys.argv[0])
        sys.exit(1)

    serial_fn = sys.argv[1]

    gm = GRBLManager(serial_fn)
    print(
        """
        q = quit
        r = reset
        bounce = bounce
        s = status
        c = calibrate
        e = spiral
        X,Y = move to X,Y
    """
    )
    for line in sys.stdin:
        line = line.strip()
        if line == "q":
            sys.exit(1)
        elif line == "r":
            r = gm.push_reset()
            print(r)
        elif line == "bounce":
            # point=np.array([2491.49001749,2401.75483327])
            # direction=np.array([0.63471637,0.57157117])
            gm.bounce(20000)

        elif line == "s":
            p = gm.update_status()
            print(p)
        elif line == "c":
            gm.calibrate()
        elif line == "e":
            gm.spiral()
        else:
            if True:
                p_main = np.array([float(x) for x in line.split()])
                a_motor_steps, b_motor_steps = gm.to_steps(p_main)
                cmd = "G0 X%0.2f Y%0.2f" % (b_motor_steps, a_motor_steps)
                print(cmd)
                print(gm.from_steps(a_motor_steps, b_motor_steps))
                gm.s.write((cmd + "\n").encode())  # Send g-code block to grbl
                grbl_out = (
                    gm.s.readline()
                )  # Wait for grbl response with carriage return
                print("MAIN OUT", grbl_out)
        time.sleep(0.01)

    gm.close()

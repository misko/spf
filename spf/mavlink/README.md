# MAVLink subsystem

Goal of this subsystem is to provide an interface for the onboard computer to control motion through ardupilot VIA (MAVlink)

## Development

### Simulator

Use windows machine to run Mission Planner, then start rover simulator, setup mavlink mirror (setup->advanced) to forward traffic to a network IP over UDP. MAKE SURE WRITE ACCESS CHECK BOX IS CLICKED!!!

For example forward to 192.168.1.139 UDP port 14551

### MAV link proxy

```
mavproxy.py --master=udp:192.168.1.139:14551 --out 127.0.0.1:14550 --out 192.168.1.139:14552
mavproxy.py --master=tcp:192.168.1.127:14560 --out 127.0.0.1:14550 --out 127.0.0.1:14552
mavproxy.py --master=tcp:192.168.1.127:14560 --out 127.0.0.1:14550 --out 127.0.0.1:14552
```

#### Mavlink controller script

This script takes in a listening IP (or serial interface , live on rover) that it uses to connect to the drone VIA MAVLink. Once connected the script uses the same BouncePlanner from the WallArray to bounce the drone in a convex boundary defined by GPS.

Youtube [link](https://youtu.be/b0P2JzziI_M)

```
python mavlink_controller.py --ip 192.168.1.139
```
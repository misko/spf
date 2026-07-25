# Rover 3 one-radio USB-IIO Zarr capture

- Frames requested: 100
- Frames stored: 100
- Signal shape: `(100, 2, 524288)`
- Signal type: `complex64`
- Logical signal data: 800 MiB
- LMDB file size after shrink: 154,066,944 bytes (147 MiB allocated)
- Collector duration: 51.3 seconds between the first and last frame timestamps
- Collector progress result: 1.93 frames/s including startup
- Median frame interval: 486.0 ms
- Median steady frame rate: 2.058 frames/s
- Median logical signal rate: 16.46 MiB/s
- Frame-interval p90: 520.3 ms
- Frame-interval p99: 652.7 ms
- Maximum frame interval: 3.942 s (initial radio reset/startup)
- All IQ values finite: pass
- Nonzero IQ present in every checked ten-frame block: pass

The capture used the committed
`rover_single_receiver_config_pi_3mhz.yaml` configuration through
`mavlink_radio_collection.py`, `DroneDataCollectorRaw`, USB-IIO, and the normal
v4 LMDB-backed Zarr writer.

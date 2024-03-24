# PlutoPlus SDR

## Introduction 

The PlutoPlus SDR [ link ](https://www.youtube.com/watch?v=higdhj46aUk) is a Adalm-Pluto like device based on the AD9361 [ link ](https://www.analog.com/en/products/ad9361.html). PlutoPlus has two coherent TX and RX ports. This allows the PlutoPlus to simultaneously sample from both RX ports and use the output to measure phase difference of an incoming signal.

## Interface

To interface with PlutoPlus SDR you can use libiio [ link ](https://github.com/analogdevicesinc/libiio) or the python wrapper (PyAdi-IIO) [ link ](https://wiki.analog.com/resources/tools-software/linux-software/pyadi-iio). 

## Issues

Sometimes when an emitter is turned on and set to broadcast, it fails to do so. For this reason any emitter used in this project must be paired with two receivers that verify the emitter came online through receiving a pilot tone. 

Even though the two RX ports on a PlutoPlus are coherent and sampled simultaneously there seems to be a half pi phase difference between the two RX ports. For this reason whenever a pair of receivers from a single SDR is brought online they must be paired with an emitter 

## Implementation

* setup_rxtx(receiver,emitter) 

Turn on the two RX ports on receiver SDR and TX0 on emitter SDR, then verify the pilot tone is received on the RX side. If the RX side does not hear the pilot tone, try again.

This returns open objects to both receiver and emitter SDR. The emitter is left in the pilot tone transmitting state.

* setup_rxtx_and_phase_calibration(receiver, emitter)

Turn on the receiver + emitter SDRs using setup_rxtx. Then assuming the emitter is equidistant from both RX[0,1] ports, calculate the phase difference between RX0 and RX1. If the phase difference is too noisy (std>0.01), try again.

This returns open objects to both receiver and emitter SDR. The emitter is not actively transmitting and the receiver SDR object has a phase_calibration field set.


## Benchmarking

[ Laptop / rpi4 ](https://docs.google.com/spreadsheets/d/1kEzWVTT2jg84SchoqwFrf9tJ0UjC-6dh6JA90qW3eL0/edit?usp=sharing)
```
python spf/sdrpluto/benchmark.py --uri ip:192.168.1.17 --buffer-sizes '2**16' '2**18' '2**20' --rx-buffers 2 --write-to-file testdata  --chunk-size 512 1024 4096 --compress blosc1 blosc4  none zstd1 zstd4
```


## Command line client


### TX only

```python sdr_controller.py --emitter-ip 192.168.1.15 --mode tx --receiver-ip 192.168.1.15```

Turn the emitter on for SDR 192.168.1.15 and leave it running after verifying its working from 192.168.1.15 (RX ports)


### RX only (self)

```python sdr_controller.py --receiver-ip 192.168.1.17 --emitter-ip 192.168.1.17 --mode rx```

Turn on the receiver ports for SDR 192.168.1.17 and plot the signal received from each channel

### RX only (other)

```python sdr_controller.py --receiver-ip 192.168.1.18 --emitter-ip 192.168.1.17 --mode rx```

Turn on the receiver ports for SDR 192.168.1.18 , using the emitter from SDR 192.168.1.17 to send pilot tone and plot the signal received from each channel

### RX calibration (other)

```python sdr_controller.py --receiver-ip 192.168.1.18 --emitter-ip 192.168.1.17 --mode rxcal ```

Turn on the receiver ports for SDR 192.168.1.18 and perform phase calibration assumping that the emitter [TX0] on 192.168.1.17 is equidistance from both RX[0,1] on SDR 192.168.1.18
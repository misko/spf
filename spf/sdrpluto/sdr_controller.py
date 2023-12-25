import argparse
import adi
import numpy as np
from spf.rf import beamformer
from math import gcd
import matplotlib.pyplot as plt
import time
import sys
from typing import Optional

import signal
import sys


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    sys.exit(0)


# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# signal.pause()

c = 3e8


class AdiWrap(adi.ad9361):
    def __init__(self, uri):
        print("%s: OPEN" % uri)
        super(AdiWrap, self).__init__(uri)

    def close(self):
        close_tx(self)
        print("%s: CLOSE" % self.uri)

    def __del__(self):
        close_tx(self)
        print("%s: DELETE!" % self.uri)


"""
Close pluto
"""


def close_tx(sdr):
    sdr.tx_enabled_channels = []
    sdr.tx_hardwaregain_chan0 = -80
    sdr.tx_hardwaregain_chan1 = -80
    sdr.tx_destroy_buffer()


class ReceiverConfig:
    def __init__(
        self,
        lo: int,
        rf_bandwidth: int,
        sample_rate: int,
        buffer_size: Optional[int] = None,
        gains: list[int] = [-30, -30],
        gain_control_mode: str = "slow_attack",
        enabled_channels: list[int] = [0, 1],
    ):
        self.lo = lo
        self.rf_bandwidth = rf_bandwidth
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.gains = gains
        self.gain_control_mode = gain_control_mode
        self.enabled_channels = enabled_channels


class EmitterConfig:
    def __init__(
        self,
        lo: int,
        rf_bandwidth: int,
        sample_rate: int,
        buffer_size: Optional[int] = None,
        gains: list = [-30, -80],
        enabled_channels: list[int] = [0],
        cyclic: bool = True,
    ):
        self.lo = lo
        self.rf_bandwidth = rf_bandwidth
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.gains = gains
        self.enabled_channels = enabled_channels
        self.cyclic = cyclic


def args_to_receiver_config(args):
    return ReceiverConfig(
        lo=args.fc,
        rf_bandwidth=int(3 * args.fi),
        sample_rate=int(args.fs),
        gains=[-30, -30],
        gain_control_mode=args.rx_mode,
        enabled_channels=[0, 1],
        buffer_size=int(args.rx_n),
    )


def args_to_emitter_config(args):
    return EmitterConfig(
        lo=args.fc,
        rf_bandwidth=int(3 * args.fi),
        sample_rate=int(args.fs),
        gains=[-30, -80],
        enabled_channels=[0],
        cyclic=True,
    )


"""
Setup the Rx part of the pluto
"""


def setup_rx(sdr, receiver_config):
    sdr.sample_rate = receiver_config.sample_rate
    assert sdr.sample_rate == receiver_config.sample_rate

    sdr.rx_rf_bandwidth = receiver_config.rf_bandwidth
    sdr.rx_lo = receiver_config.lo

    # setup the gain mode
    sdr.rx_hardwaregain_chan0 = receiver_config.gains[0]
    sdr.rx_hardwaregain_chan1 = receiver_config.gains[1]
    sdr.gain_control_mode = receiver_config.gain_control_mode

    if receiver_config.buffer_size is not None:
        sdr.rx_buffer_size = receiver_config.buffer_size
    # sdr._rxadc.set_kernel_buffers_count(
    #    1
    # )  # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
    sdr.rx_enabled_channels = receiver_config.enabled_channels

    # turn off TX
    sdr.tx_enabled_channels = []
    sdr.tx_destroy_buffer()


"""
Setup the Tx side of the pluto
"""


def setup_tx(sdr, emitter_config):
    sdr.tx_cyclic_buffer = emitter_config.cyclic  # this keeps repeating!

    sdr.sample_rate = emitter_config.sample_rate
    assert sdr.sample_rate == emitter_config.sample_rate

    sdr.tx_rf_bandwidth = emitter_config.rf_bandwidth
    sdr.tx_lo = emitter_config.lo

    # setup the gain mode
    sdr.tx_hardwaregain_chan0 = emitter_config.gains[0]
    sdr.tx_hardwaregain_chan1 = emitter_config.gains[1]

    if emitter_config.buffer_size is not None:
        sdr.tx_buffer_size = emitter_config.buffer_size
    # sdr._rxadc.set_kernel_buffers_count(
    #    1
    # )  # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
    sdr.tx_enabled_channels = emitter_config.enabled_channels


def make_tone(args):
    # create a buffe for the signal
    fc0 = int(args.fi)
    fs = int(args.fs)  # must be <=30.72 MHz if both channels are enabled
    tx_n = int(fs / gcd(fs, fc0))
    while tx_n < 1024 * 16:
        tx_n *= 2

    # since its a cyclic buffer its important to end on a full phase
    t = (
        np.arange(0, tx_n) / fs
    )  # time at each point assuming we are sending samples at (1/fs)s
    return np.exp(1j * 2 * np.pi * fc0 * t) * (2**14)


"""
Given an online SDR receiver check if the max power peak is as expected during calibration
"""


def check_for_freq_peak(sdr, args):
    freq = np.fft.fftfreq(args.rx_n, d=1.0 / args.fs)
    signal_matrix = np.vstack(sdr.rx())
    sp = np.fft.fft(signal_matrix[0])
    max_freq = freq[np.abs(np.argmax(sp.real))]
    if np.abs(max_freq - args.fi) < (args.fs / args.rx_n + 1):
        return True
    return False


def setup_rxtx(receiver_uri, receiver_config, emitter_uri, emitter_config):
    retries = 0
    while retries < 10:
        # sdr_rx = adi.ad9361(uri=receiver_uri)
        sdr_rx = AdiWrap(uri=receiver_uri)
        setup_rx(sdr_rx, receiver_config)

        sdr_tx = sdr_rx
        if receiver_uri != emitter_uri:
            # sdr_tx = adi.ad9361(uri=emitter_uri)
            sdr_tx = AdiWrap(uri=emitter_uri)

        setup_tx(sdr_tx, emitter_config)

        # start TX
        sdr_tx.tx(make_tone(args))

        # get RX and drop it
        for _ in range(40):
            sdr_rx.rx()

        # test to see what frequency we are seeing
        if check_for_freq_peak(sdr_rx, args):
            # close_tx(sdr_tx)
            return sdr_rx, sdr_tx
        retries += 1

        # try to reset
        sdr_tx.close()
        sdr_rx = None
        sdr_tx = None
        time.sleep(1)
    return None, None


"""
Use a single PlutoSDR receiver_ip to calibrate phase
Using Tx1 to emit and Rx1 + Rx2 to receive 
"""


def setup_rxtx_and_phase_calibration(
    receiver_uri, receiver_config, emitter_uri, emitter_config
):
    print("%s: Starting inter antenna receiver phase calibration" % receiver_uri)

    # its important to not use the emitter uri when calibrating!
    sdr_rx, sdr_tx = setup_rxtx(
        receiver_uri=receiver_uri,
        receiver_config=receiver_config,
        emitter_uri=emitter_uri,
        emitter_config=emitter_config,
    )

    if sdr_rx is None:
        print("Failed to bring rx tx online")
        return None
    print("%s: Emitter online verified by %s" % (emitter_uri, receiver_uri))

    # sdr_rx.phase_calibration=0
    # return sdr_rx,sdr_tx
    # get some new data
    print(
        "%s: Starting phase calibration (using emitter: %s)"
        % (receiver_uri, emitter_uri)
    )
    for retry in range(20):
        n_calibration_frames = 800
        phase_calibrations = np.zeros(n_calibration_frames)
        for idx in range(n_calibration_frames):
            signal_matrix = np.vstack(sdr_rx.rx())
            phase_calibrations[idx] = (
                (np.angle(signal_matrix[0]) - np.angle(signal_matrix[1])) % (2 * np.pi)
            ).mean()  # TODO THIS BREAKS if diff is near 2*np.pi...
        print(
            "%s: Phase calibration mean (%0.4f) std (%0.4f)"
            % (args.receiver_ip, phase_calibrations.mean(), phase_calibrations.std())
        )
        if phase_calibrations.std() < 0.01:
            sdr_tx.close()
            print(
                "%s: Final phase calibration (radians) is %0.4f"
                % (args.receiver_ip, phase_calibrations.mean()),
                "(fraction of 2pi) %0.4f" % (phase_calibrations.mean() / (2 * np.pi)),
            )
            sdr_rx.phase_calibration = phase_calibrations.mean()
            return sdr_rx, sdr_tx
    sdr_tx.close()
    print("%s: Phase calibration failed" % args.receiver_ip)
    return None


def circular_mean(angles, trim=50.0):
    cm = np.arctan2(np.sin(angles).sum(), np.cos(angles).sum()) % (2 * np.pi)
    dists = np.vstack([2 * np.pi - np.abs(cm - angles), np.abs(cm - angles)]).min(
        axis=0
    )
    _angles = angles[dists < np.percentile(dists, 100.0 - trim)]
    _cm = np.arctan2(np.sin(_angles).sum(), np.cos(_angles).sum()) % (2 * np.pi)
    return cm, _cm


def get_avg_phase(signal_matrix, trim=0.0):
    # signal_matrix=np.vstack(sdr_rx.rx())
    # signal_matrix[1]*=np.exp(1j*sdr_rx.phase_calibration)

    diffs = (np.angle(signal_matrix[0]) - np.angle(signal_matrix[1])) % (2 * np.pi)
    mean, _mean = circular_mean(diffs, trim=50.0)

    return mean, _mean


def plot_recv_signal(sdr_rx):
    pos = np.array([[-0.03, 0], [0.03, 0]])
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))

    rx_n = sdr_rx.rx_buffer_size
    t = np.arange(rx_n)
    while True:
        signal_matrix = np.vstack(sdr_rx.rx())
        signal_matrix[1] *= np.exp(1j * sdr_rx.phase_calibration)

        beam_thetas, beam_sds, beam_steer = beamformer(pos, signal_matrix, args.fc)

        freq = np.fft.fftfreq(t.shape[-1], d=1.0 / sdr_rx.sample_rate)
        assert t.shape[-1] == rx_n
        for idx in [0, 1]:
            axs[idx][0].clear()
            axs[idx][1].clear()

            axs[idx][0].scatter(t, signal_matrix[idx].real, s=1)
            axs[idx][0].set_xlabel("Time")
            axs[idx][0].set_ylabel("Real(signal)")
            axs[idx][0].set_ylim([-1000, 1000])

            sp = np.fft.fft(signal_matrix[idx])
            axs[idx][1].scatter(freq, np.log(np.abs(sp.real)), s=1)  # , freq, sp.imag)
            axs[idx][1].set_xlabel("Frequency bin")
            axs[idx][1].set_ylabel("Power")
            axs[idx][1].set_ylim([-30, 30])
            max_freq = freq[np.abs(np.argmax(sp.real))]
            axs[idx][1].axvline(x=max_freq, label="max %0.2e" % max_freq, color="red")
            axs[idx][1].legend()

            axs[idx][2].clear()
            axs[idx][2].scatter(signal_matrix[idx].real, signal_matrix[idx].imag, s=1)
            axs[idx][2].set_xlabel("I real(signal)")
            axs[idx][2].set_ylabel("Q imag(signal)")
            axs[idx][2].set_title("IQ plot recv (%d)" % idx)
            axs[idx][2].set_ylim([-600, 600])
            axs[idx][2].set_xlim([-600, 600])

            axs[idx][0].set_title("Real signal recv (%d)" % idx)
            axs[idx][1].set_title("Power recv (%d)" % idx)
            # print("MAXFREQ",freq[np.abs(np.argmax(sp.real))])
        diff = (np.angle(signal_matrix[0]) - np.angle(signal_matrix[1])) % (2 * np.pi)
        axs[0][3].clear()
        axs[0][3].scatter(t, diff, s=1)
        mean, _mean = circular_mean(diff)
        # print(mean,_mean)
        axs[0][3].axhline(y=mean, color="black", label="circular mean")
        axs[0][3].axhline(y=_mean, color="red", label="trimmed circular mean")
        axs[0][3].set_ylim([0, 2 * np.pi])
        axs[0][3].set_xlabel("Time")
        axs[0][3].set_ylabel("Angle estimate")
        axs[0][3].legend()
        axs[0][3].set_title("Angle estimate")
        axs[1][3].clear()
        axs[1][3].plot(beam_thetas, beam_sds)

        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.00001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receiver-ip",
        type=str,
        help="Receivers",
        required=True,
    )
    parser.add_argument(
        "--emitter-ip", type=str, help="Long term emitter", required=True
    )
    parser.add_argument("--mode", choices=["rx", "rxcal", "tx"], required=True)
    parser.add_argument(
        "--fi",
        type=int,
        help="Intermediate frequency",
        required=False,
        default=int(1e5),
    )
    parser.add_argument(
        "--fc",
        type=int,
        help="Carrier frequency",
        required=False,
        default=int(2.5e9),
    )
    parser.add_argument(
        "--fs",
        type=int,
        help="Sampling frequency",
        required=False,
        default=int(16e6),
    )
    parser.add_argument(
        "--cal0",
        type=int,
        help="Rx0 calibration phase offset in degrees",
        required=False,
        default=180,
    )
    parser.add_argument(
        "--rx-gain", type=int, help="RX gain", required=False, default=-3
    )
    parser.add_argument(
        "--tx-gain", type=int, help="TX gain", required=False, default=-8
    )
    parser.add_argument(
        "--rx-mode",
        type=str,
        help="rx mode",
        required=False,
        default="slow_attack",
        choices=["manual", "slow_attack", "fast_attack"],
    )
    parser.add_argument(
        "--rx-n",
        type=int,
        help="RX buffer size",
        required=False,
        default=int(2**9),
    )  # 12
    args = parser.parse_args()

    # calibrate the receiver

    emitter_uri = "ip:%s" % args.emitter_ip
    receiver_uri = "ip:%s" % args.receiver_ip

    if args.mode == "rxcal":
        # if we use weaker tx gain then the noise in phase calibration goes up
        emitter_config = args_to_emitter_config(args)
        emitter_config.gains = [-30, -80]

        sdr_rx, sdr_tx = setup_rxtx_and_phase_calibration(
            receiver_uri=receiver_uri,
            receiver_config=args_to_receiver_config(args),
            emitter_uri=emitter_uri,
            emitter_config=emitter_config,
        )

        if sdr_rx is None:
            print("Failed phase calibration, exiting")
            sys.exit(1)

        plot_recv_signal(sdr_rx)

    elif args.mode == "rx":
        # sdr_rx = adi.ad9361(uri=receiver_uri)
        sdr_rx = AdiWrap(uri=receiver_uri)
        receiver_config = args_to_receiver_config(args)
        setup_rx(sdr_rx, receiver_config)
        sdr_rx.phase_calibration = args.cal0
        plot_recv_signal(sdr_rx)
    elif args.mode == "tx":
        sdr_rx, sdr_tx = setup_rxtx(
            receiver_uri=emitter_uri,
            receiver_config=args_to_receiver_config(args),
            emitter_uri=emitter_uri,
            emitter_config=args_to_emitter_config(args),
        )
        if sdr_rx is None:
            print("Failed to bring emitter online")
            sys.exit(1)
        print("%s: Emitter online verified by %s" % (emitter_uri, receiver_uri))
        # apply the previous calibration
        time.sleep(600)

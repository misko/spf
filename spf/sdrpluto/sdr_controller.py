import argparse
import logging
import sys
import time
from datetime import datetime
from math import gcd
from typing import Optional

import adi
import matplotlib.pyplot as plt
import numpy as np

from spf.rf import ULADetector, beamformer

# TODO close SDR on exit
# import signal
# def signal_handler(sig, frame):
#    print("You pressed Ctrl+C!")
#    sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# signal.pause()

c = 3e8

pplus_online = {}

run_radios = True


def shutdown_radios():
    global run_radios
    run_radios = False


def get_uri(rx_config=None, tx_config=None, uri=None):
    assert rx_config is not None or tx_config is not None or uri is not None
    if rx_config is not None and tx_config is not None:
        assert rx_config.uri == tx_config.uri
    if rx_config is not None:
        return rx_config.uri
    elif tx_config is not None:
        return tx_config.uri
    return uri


# TODO not thread safe
def get_pplus(rx_config=None, tx_config=None, uri=None):
    uri = get_uri(rx_config=rx_config, tx_config=tx_config, uri=uri)
    global pplus_online
    if uri not in pplus_online:
        pplus_online[uri] = PPlus(rx_config=rx_config, tx_config=tx_config, uri=uri)
    else:
        pplus_online[uri].set_config(rx_config=rx_config, tx_config=tx_config)
    logging.debug(f"{uri}: get_pplus PlutoPlus")
    return pplus_online[uri]


class PPlus:
    def __init__(self, rx_config=None, tx_config=None, uri=None):
        super(PPlus, self).__init__()
        self.uri = get_uri(rx_config=rx_config, tx_config=tx_config, uri=uri)
        logging.info(f"{self.uri}: Open PlutoPlus")

        # try to fix issue with radios coming online
        self.sdr = adi.ad9361(uri=self.uri)
        self.close_tx()
        # self.sdr = None
        time.sleep(0.1)

        # open for real
        # self.sdr = adi.ad9361(uri=self.uri)
        self.tx_config = None
        self.rx_config = None
        self.set_config(rx_config=rx_config, tx_config=tx_config)

        if (
            self.tx_config is None and self.rx_config is None
        ):  # this is a fresh open or reset
            self.close_tx()
            self.sdr.tx_destroy_buffer()
            self.sdr.rx_destroy_buffer()
            self.sdr.tx_enabled_channels = []

    def set_config(self, rx_config=None, tx_config=None):
        logging.debug(f"{self.uri}: set_config RX{str(rx_config)} TX{str(tx_config)})")
        # RX should be setup like this
        if rx_config is not None:
            assert self.rx_config is None
            self.rx_config = rx_config

        # TX should be setup like this
        if tx_config is not None:
            assert self.tx_config is None
            self.tx_config = tx_config

    def close(self):
        logging.info(f"{self.uri}: Start close PlutoPlus")
        self.close_tx()
        logging.info(f"{self.uri}: Done close PlutoPlus")

    def __del__(self):
        logging.debug(f"{self.uri}: Start delete PlutoPlus")
        self.close_tx()
        self.sdr.tx_destroy_buffer()
        self.sdr.rx_destroy_buffer()
        self.sdr.tx_enabled_channels = []
        logging.debug(f"{self.uri}: Done delete PlutoPlus")

    """
    Setup the Rx part of the pluto
    """

    def setup_rx(self):
        self.sdr.sample_rate = self.rx_config.sample_rate
        assert self.sdr.sample_rate == self.rx_config.sample_rate

        self.sdr.rx_rf_bandwidth = self.rx_config.rf_bandwidth
        self.sdr.rx_lo = self.rx_config.lo

        # setup the gain mode
        self.sdr.rx_hardwaregain_chan0 = self.rx_config.gains[0]
        self.sdr.rx_hardwaregain_chan1 = self.rx_config.gains[1]
        self.sdr.gain_control_mode = self.rx_config.gain_control_mode

        if self.rx_config.buffer_size is not None:
            self.sdr.rx_buffer_size = self.rx_config.buffer_size
        self.sdr._rxadc.set_kernel_buffers_count(
            self.rx_config.rx_buffers
        )  # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
        self.sdr.rx_enabled_channels = self.rx_config.enabled_channels

    """
    Setup the Tx side of the pluto
    """

    def setup_tx(self):
        logging.debug(f"{self.tx_config.uri}: Setup TX")
        self.sdr.tx_destroy_buffer()
        self.sdr.tx_cyclic_buffer = self.tx_config.cyclic  # this keeps repeating!

        self.sdr.sample_rate = self.tx_config.sample_rate
        assert self.sdr.sample_rate == self.tx_config.sample_rate

        self.sdr.tx_rf_bandwidth = self.tx_config.rf_bandwidth
        self.sdr.tx_lo = self.tx_config.lo

        # setup the gain mode
        self.sdr.tx_hardwaregain_chan0 = self.tx_config.gains[0]
        self.sdr.tx_hardwaregain_chan1 = self.tx_config.gains[1]

        if self.tx_config.buffer_size is not None:
            self.sdr.tx_buffer_size = self.tx_config.buffer_size
        # self.sdr._rxadc.set_kernel_buffers_count(
        #    1
        # )  # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
        self.sdr.tx_enabled_channels = self.tx_config.enabled_channels

    """
    Close pluto
    """

    def close_tx(self):
        self.sdr.tx_hardwaregain_chan0 = -80
        self.sdr.tx_hardwaregain_chan1 = -80
        self.sdr.tx_enabled_channels = []
        self.sdr.tx_destroy_buffer()
        self.tx_config = None
        time.sleep(0.1)

    def close_rx(self):
        self.rx_config = None

    """
    Given an online SDR receiver check if the max power peak is as expected during calibration
    """

    def check_for_freq_peak(self):
        freq = np.fft.fftfreq(
            self.rx_config.buffer_size, d=1.0 / self.rx_config.sample_rate
        )
        signal_matrix = np.vstack(self.sdr.rx())
        sp = np.fft.fft(signal_matrix[0])
        max_freq = freq[np.abs(np.argmax(sp.real))]
        if np.abs(max_freq - self.rx_config.intermediate) < (
            self.rx_config.sample_rate / self.rx_config.buffer_size + 1
        ):
            return True
        return False


class Config:
    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="\n".join(
                "{}={!r}".format(k, v) for k, v in sorted(self.__dict__.items())
            ),
        )


class ReceiverConfig(Config):
    def __init__(
        self,
        lo: int,
        rf_bandwidth: int,
        sample_rate: int,
        intermediate: int,
        uri: str,
        buffer_size: Optional[int] = None,
        gains: list[int] = [-30, -30],
        gain_control_mode: str = "slow_attack",
        enabled_channels: list[int] = [0, 1],
        rx_spacing=None,
        rx_theta_in_pis=0.0,
        motor_channel=None,
        rx_buffers=4,
    ):
        self.lo = lo
        self.rf_bandwidth = rf_bandwidth
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.gains = gains
        self.gain_control_mode = gain_control_mode
        self.enabled_channels = enabled_channels
        self.intermediate = intermediate
        self.uri = uri
        self.rx_spacing = rx_spacing
        self.rx_theta_in_pis = rx_theta_in_pis
        self.motor_channel = motor_channel
        self.rx_buffers = rx_buffers

        if self.rx_spacing is not None:
            self.rx_pos = ULADetector(
                sampling_frequency=None,
                n_elements=2,
                spacing=self.rx_spacing,
                orientation=0.0,
            ).all_receiver_pos()
            self.rx_pos_rotated = ULADetector(
                sampling_frequency=None,
                n_elements=2,
                spacing=self.rx_spacing,
                orientation=self.rx_theta_in_pis * np.pi,
            ).all_receiver_pos()

            logging.info(
                f"{self.uri}:RX antenna positions (theta_in_pis:{self.rx_theta_in_pis}):"
            )
            logging.info(f"{self.uri}:\tRX[0]:{str(self.rx_pos[0])}")
            logging.info(f"{self.uri}:\tRX[1]:{str(self.rx_pos[1])}")
        else:
            self.rx_pos = None


class EmitterConfig(Config):
    def __init__(
        self,
        lo: int,
        rf_bandwidth: int,
        sample_rate: int,
        intermediate: int,
        uri: str,
        buffer_size: Optional[int] = None,
        gains: list = [-30, -80],
        enabled_channels: list[int] = [0],
        cyclic: bool = True,
        motor_channel: int = None,
    ):
        self.lo = lo
        self.rf_bandwidth = rf_bandwidth
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.gains = gains
        self.enabled_channels = enabled_channels
        self.cyclic = cyclic
        self.intermediate = intermediate
        self.uri = uri
        self.motor_channel = motor_channel


def args_to_rx_config(args):
    return ReceiverConfig(
        lo=args.fc,
        rf_bandwidth=int(3 * args.fi),
        sample_rate=int(args.fs),
        gains=[-30, -30],
        gain_control_mode=args.rx_mode,
        enabled_channels=[0, 1],
        buffer_size=int(args.rx_n),
        intermediate=args.fi,
        uri="ip:%s" % args.receiver_ip,
        rx_spacing=args.rx_spacing,
        # rx_theta_in_pis=0.25,
    )


def args_to_tx_config(args):
    return EmitterConfig(
        lo=args.fc,
        rf_bandwidth=int(3 * args.fi),
        sample_rate=int(args.fs),
        gains=[args.tx_gain, -80],
        enabled_channels=[0],
        cyclic=True,
        intermediate=args.fi,
        uri="ip:%s" % args.emitter_ip,
    )


def make_tone(tx_config: EmitterConfig):
    # create a buffe for the signal
    fc0 = tx_config.intermediate
    fs = tx_config.sample_rate  # must be <=30.72 MHz if both channels are enabled
    tx_n = int(fs / gcd(fs, fc0))
    while tx_n < 1024 * 16:
        tx_n *= 2

    # since its a cyclic buffer its important to end on a full phase
    t = (
        np.arange(0, tx_n) / fs
    )  # time at each point assuming we are sending samples at (1/fs)s
    return np.exp(1j * 2 * np.pi * fc0 * t) * (2**14)


def setup_rxtx(rx_config, tx_config, leave_tx_on=False, provided_pplus_rx=None):
    retries = 0
    while run_radios and retries < 15:
        logging.info(f"setup_rxtx({rx_config.uri}, {tx_config.uri}) retry {retries}")
        # sdr_rx = adi.ad9361(uri=receiver_uri)
        if provided_pplus_rx is None:
            if rx_config.uri == tx_config.uri:
                logging.debug(f"{rx_config.uri} RX TX are same")
                pplus_rx = get_pplus(rx_config=rx_config, tx_config=tx_config)
                pplus_tx = pplus_rx
            else:
                logging.debug(f"{rx_config.uri}(RX) TX are different")
                pplus_rx = get_pplus(rx_config=rx_config)
                logging.debug(f"{tx_config.uri} RX (TX) are different")
                pplus_tx = get_pplus(tx_config=tx_config)
            pplus_rx.setup_rx()
        else:
            if rx_config.uri == tx_config.uri:
                logging.debug(f"{rx_config.uri} RX TX are same")
                pplus_tx = provided_pplus_rx
            else:
                logging.debug(f"{tx_config.uri} RX (TX) are different")
                pplus_tx = get_pplus(tx_config=tx_config)
            # TODO if pluto_rx is provided confirm its the same config
            pplus_rx = provided_pplus_rx

        pplus_tx.setup_tx()
        time.sleep(0.1)

        # start TX
        pplus_tx.sdr.tx(make_tone(tx_config))
        time.sleep(0.1)

        # get RX and drop it
        for _ in range(40):
            pplus_rx.sdr.rx()

        # test to see what frequency we are seeing
        if pplus_rx.check_for_freq_peak():
            logging.info(
                f"RXTX came online with configs\nRX_config:{pplus_rx.rx_config}\nTX_config:{pplus_tx.tx_config}"
            )
            if not leave_tx_on:
                pplus_tx.close_tx()
            return pplus_rx, pplus_tx
        if provided_pplus_rx is None:
            pplus_rx.close_rx()
        pplus_tx.close_tx()
        retries += 1

        # try to reset
        pplus_tx.close_tx()
        time.sleep(0.1)
    return None, None


"""
Use a single PlutoSDR receiver_ip to calibrate phase
Using Tx1 to emit and Rx1 + Rx2 to receive 
"""


def setup_rxtx_and_phase_calibration(
    rx_config: ReceiverConfig,
    tx_config: EmitterConfig,
    tolerance=0.01,
    n_calibration_frames=800,
    leave_tx_on=False,
    using_tx_already_on=None,
):
    logging.info(f"{rx_config.uri}: Starting inter antenna receiver phase calibration")

    # its important to not use the emitter uri when calibrating!
    if using_tx_already_on is not None:
        logging.debug(f"{rx_config.uri}: TX already on!")
        pplus_rx = get_pplus(rx_config=rx_config)
        pplus_rx.setup_rx()
        pplus_tx = using_tx_already_on
    else:
        logging.debug(f"{rx_config.uri}: TX not on!")
        pplus_rx, pplus_tx = setup_rxtx(
            rx_config=rx_config, tx_config=tx_config, leave_tx_on=True
        )

    if pplus_rx is None:
        logging.info(f"{rx_config.uri}: Failed to bring rx tx online")
        return None, None
    logging.info(f"{tx_config.uri}: TX online verified by RX {rx_config.uri}")

    # sdr_rx.phase_calibration=0
    # return sdr_rx,sdr_tx
    # get some new data
    logging.info(
        f"{rx_config.uri}: Starting phase calibration (using emitter: {tx_config.uri})"
    )
    retries = 0
    while run_radios and retries < 20:
        logging.debug(f"{rx_config.uri} RETRY {retries}")
        phase_calibrations = np.zeros(n_calibration_frames)
        phase_calibrations_cm = np.zeros(n_calibration_frames)
        for idx in range(n_calibration_frames):
            if not run_radios:
                break
            signal_matrix = np.vstack(pplus_rx.sdr.rx())
            phase_calibrations[idx] = (
                (np.angle(signal_matrix[0]) - np.angle(signal_matrix[1])) % (2 * np.pi)
            ).mean()  # TODO THIS BREAKS if diff is near 2*np.pi...
            phase_calibrations_cm[idx], _ = circular_mean(
                np.angle(signal_matrix[0]) - np.angle(signal_matrix[1])
            )
        phase_calibration_u = phase_calibrations.mean()
        phase_calibration_std = phase_calibrations.std()
        logging.info(
            f"{rx_config.uri}: Phase calibration mean \
                ({phase_calibration_u:0.4f}) std ({phase_calibration_std:0.4f})"
        )

        # TODO this part should also get replaced by circular mean
        phase_calibration_cm_u = circular_mean(phase_calibrations_cm)[0]
        phase_calibration_cm_std = phase_calibrations_cm.std()
        logging.info(
            f"{rx_config.uri}: Phase calibration mean CM \
                ({phase_calibration_cm_u:0.4f}) std ({phase_calibration_cm_std:0.4f})"
        )

        if phase_calibration_std < tolerance:
            if not leave_tx_on:
                pplus_tx.close()
            logging.info(
                f"{rx_config.uri}: Final phase calibration (radians) is {phase_calibration_u:0.4f}\
                 (fraction of 2pi) {(phase_calibration_u / (2 * np.pi)):0.4f}"
            )
            pplus_rx.phase_calibration = phase_calibrations.mean()
            return pplus_rx, pplus_tx
    pplus_tx.close()
    logging.error(f"{rx_config.uri}: Phase calibration failed")
    return None, None


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


def plot_recv_signal(pplus_rx):
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))

    rx_n = pplus_rx.sdr.rx_buffer_size
    t = np.arange(rx_n)
    while run_radios:
        signal_matrix = np.vstack(pplus_rx.sdr.rx())
        signal_matrix[1] *= np.exp(1j * pplus_rx.phase_calibration)
        assert pplus_rx.rx_config.rx_pos is not None
        beam_thetas, beam_sds, _ = beamformer(
            pplus_rx.rx_config.rx_pos, signal_matrix, pplus_rx.rx_config.lo
        )

        freq = np.fft.fftfreq(t.shape[-1], d=1.0 / pplus_rx.sdr.sample_rate)
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
        diff = (np.angle(signal_matrix[0]) - np.angle(signal_matrix[1])) % (2 * np.pi)
        axs[0][3].clear()
        axs[0][3].scatter(t, diff, s=1)
        mean, _mean = circular_mean(diff)
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
    parser.add_argument(
        "--rx-spacing",
        type=float,
        help="RX spacing",
        required=False,
        default=0.065,
    )  # 12
    args = parser.parse_args()

    # calibrate the receiver
    # setup logging
    run_started_at = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_files_prefix = f"wallarrayv2_{run_started_at}"
    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"{output_files_prefix}.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=getattr(logging, "DEBUG", None),
    )

    emitter_uri = "ip:%s" % args.emitter_ip
    receiver_uri = "ip:%s" % args.receiver_ip

    if args.mode == "rxcal":
        # if we use weaker tx gain then the noise in phase calibration goes up
        tx_config = args_to_tx_config(args)
        tx_config.gains = [-30, -80]
        pplus_rx, pplus_tx = setup_rxtx_and_phase_calibration(
            rx_config=args_to_rx_config(args),
            tx_config=tx_config,
        )

        if pplus_rx is None:
            logging.error("Failed phase calibration, exiting")
            sys.exit(1)

        plot_recv_signal(pplus_rx)

    elif args.mode == "rx":
        pplus_rx = get_pplus(rx_config=args_to_rx_config(args))
        pplus_rx.setup_rx()
        pplus_rx.phase_calibration = args.cal0

        plot_recv_signal(pplus_rx)
    elif args.mode == "tx":
        pplus_rx, pplus_tx = setup_rxtx(
            rx_config=args_to_rx_config(args),
            tx_config=args_to_tx_config(args),
            leave_tx_on=True,
        )
        if pplus_rx is None:
            logging.error("Failed to bring emitter online")
            sys.exit(1)
        logging.info(f"{emitter_uri}: Emitter online verified by {receiver_uri}")
        # apply the previous calibration
        time.sleep(1800)

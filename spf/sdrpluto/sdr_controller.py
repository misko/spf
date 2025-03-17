import argparse
import logging
import sys
import time
from datetime import datetime
from math import gcd
from typing import List, Optional

import adi

try:
    import bladerf
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
from bladerf import _bladerf
from tqdm import tqdm

from spf.rf import (
    ULADetector,
    beamformer,
    beamformer_given_steering,
    beamformer_thetas,
    circular_mean,
    pi_norm,
    precompute_steering_vectors,
)
from spf.utils import random_signal_matrix


def bladerf_serial_to_info():
    try:
        serial_to_uri = {}
        for dev in _bladerf.get_device_list():
            dev_dict = dev._asdict()
            serial_to_uri[dev_dict["serial"].decode()] = (
                f'bladerf://libusb:device={dev_dict["usb_bus"]}:{dev_dict["usb_addr"]}'
            )
        return serial_to_uri
    except _bladerf.BladeRFError:
        print("No bladeRF devices found.")
        return {}


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
        gain_control_modes: List[str] = ["slow_attack", "slow_attack"],
        enabled_channels: list[int] = [0, 1],
        rx_spacing=None,
        rx_theta_in_pis=0.0,
        motor_channel=None,
        rx_buffers=4,
        filter_fir_en=1,
    ):
        self.lo = lo
        self.rf_bandwidth = rf_bandwidth
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.gains = gains
        self.gain_control_modes = gain_control_modes
        self.enabled_channels = enabled_channels
        self.intermediate = intermediate
        self.uri = uri
        self.rx_spacing = rx_spacing
        self.rx_theta_in_pis = rx_theta_in_pis
        self.motor_channel = motor_channel
        self.rx_buffers = rx_buffers
        self.filter_fir_en = filter_fir_en

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

            logging.debug(
                f"{self.uri}:RX antenna positions (theta_in_pis:{self.rx_theta_in_pis}):"
            )
            logging.debug(f"{self.uri}:\tRX[0]:{str(self.rx_pos[0])}")
            logging.debug(f"{self.uri}:\tRX[1]:{str(self.rx_pos[1])}")
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


def rx_config_from_receiver_yaml(receiver_yaml):
    return ReceiverConfig(
        lo=receiver_yaml["f-carrier"],
        rf_bandwidth=receiver_yaml["bandwidth"],
        sample_rate=receiver_yaml["f-sampling"],
        gains=[receiver_yaml["rx-gain"], receiver_yaml["rx-gain"]],
        gain_control_modes=[
            receiver_yaml["rx-gain-mode"],
            receiver_yaml["rx-gain-mode"],
        ],
        enabled_channels=[0, 1],
        buffer_size=receiver_yaml["buffer-size"],
        intermediate=receiver_yaml["f-intermediate"],
        uri=receiver_yaml["receiver-uri"],
        rx_spacing=receiver_yaml["antenna-spacing-m"],
        rx_theta_in_pis=receiver_yaml["theta-in-pis"],
        motor_channel=(
            receiver_yaml["motor_channel"] if "motor_channel" in receiver_yaml else None
        ),
        rx_buffers=receiver_yaml["rx-buffers"],
        filter_fir_en=(
            receiver_yaml["filter_fir_en"] if "filter_fir_en" in receiver_yaml else 1
        ),
    )


def args_to_rx_config(args):
    return ReceiverConfig(
        lo=args.fc,
        rf_bandwidth=int(3 * args.fi),
        sample_rate=int(args.fs),
        gains=[args.rx_gain, args.rx_gain],
        gain_control_modes=[args.rx_mode, args.rx_mode],
        enabled_channels=[0, 1],
        buffer_size=int(args.rx_n),
        intermediate=args.fi,
        uri=args.receiver_uri,
        rx_spacing=args.rx_spacing,
        rx_buffers=args.kernel_buffers,
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
        uri=args.emitter_uri,
    )


def shutdown_radios():
    global run_radios
    run_radios = False


def get_uri(
    rx_config: ReceiverConfig = None, tx_config: EmitterConfig = None, uri: str = None
):
    assert rx_config is not None or tx_config is not None or uri is not None
    if rx_config is not None and tx_config is not None:
        assert rx_config.uri == tx_config.uri
    if rx_config is not None:
        return rx_config.uri
    elif tx_config is not None:
        return tx_config.uri
    return uri


# TODO not thread safe
def get_pplus(
    rx_config: ReceiverConfig = None, tx_config: EmitterConfig = None, uri: str = None
):
    uri = get_uri(rx_config=rx_config, tx_config=tx_config, uri=uri)
    global pplus_online
    if uri not in pplus_online:
        if "fake" in uri:
            pplus_online[uri] = FakePPlus(
                rx_config=rx_config, tx_config=tx_config, uri=uri
            )
        elif "bladerf" in uri:
            pplus_online[uri] = BladeRFSdr(
                rx_config=rx_config, tx_config=tx_config, uri=uri
            )
        else:
            pplus_online[uri] = PPlus(rx_config=rx_config, tx_config=tx_config, uri=uri)
    else:
        pplus_online[uri].set_config(rx_config=rx_config, tx_config=tx_config)
    logging.debug(f"{uri}: get_pplus PlutoPlus")
    return pplus_online[uri]


class FakeSdr:
    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size

    def rx(self):
        time.sleep(0.3)
        return random_signal_matrix(2 * self.buffer_size).reshape(2, self.buffer_size)

    def tx(self, _):
        pass


class BladeRFSdr:
    def __init__(
        self,
        uri: str,
        rx_config: ReceiverConfig = None,
        tx_config: EmitterConfig = None,
        phase_calibration=0.0,
    ):
        super(BladeRFSdr, self).__init__()
        assert "bladerf://" in uri  # only support one device per machine

        self.tx_config = None
        self.rx_config = None

        self.uri = get_uri(rx_config=rx_config, tx_config=tx_config, uri=uri)

        if "serial" in self.uri:
            serial_to_find = self.uri.split("serial:")[1]
            bladerfs_on_system = bladerf_serial_to_info()
            if serial_to_find in bladerfs_on_system:
                self.uri = bladerfs_on_system[serial_to_find]
            else:
                raise ValueError(f"Missing bladerf {serial_to_find}")

        self.uri = self.uri[len("bladerf://") :]
        logging.info(f"{self.uri}: Open bladeRF")

        self.set_config(rx_config=rx_config, tx_config=tx_config)

        # Open the first available bladeRF device (or parse URI if needed).
        self.sdr = bladerf.BladeRF(device_identifier=self.uri)
        self.sdr.rx = self.rx  # TODO make this nicer

        self.rx_channels = (
            self.sdr.Channel(bladerf.CHANNEL_RX(0)),
            self.sdr.Channel(bladerf.CHANNEL_RX(1)),
        )
        self.tx_channels = (
            self.sdr.Channel(bladerf.CHANNEL_TX(0)),
            self.sdr.Channel(bladerf.CHANNEL_TX(1)),
        )
        for tx_ch in self.tx_channels:
            tx_ch.enable = False

    def soft_reset_radio(self):
        return

    def rssis(self):
        rssis = np.array(
            [
                self.rx_channels[0].rssi,
                self.rx_channels[1].rssi,
            ]
        )
        return rssis.mean(axis=1)
        # rssis = np.array([0, 0])
        # return rssis

    def gains(self):
        # gains = np.array([0, 0])
        gains = np.array(
            [
                self.rx_channels[0].gain,
                self.rx_channels[1].gain,
            ]
        )
        return gains

    def set_config(
        self, rx_config: ReceiverConfig = None, tx_config: EmitterConfig = None
    ):
        logging.debug(f"{self.uri}: set_config RX{str(rx_config)} TX{str(tx_config)})")
        # RX should be setup like this
        if rx_config is not None:
            assert self.rx_config is None
            self.rx_config = rx_config

        # TX should be setup like this
        if tx_config is not None:
            assert self.tx_config is None
            self.tx_config = tx_config

        assert tx_config is None

    def close(self):
        pass

    def setup_rx_config(self):
        """
        Configure the AD9361 on the bladeRF 2.0 Micro to enable:
        1) LNA bypass phase inversion correction (reg 0x022 bit6)
        2) RX1-RX2 phase alignment (reg 0x011 bit5)
        3) Clear bit5 in reg 0x189 to disable inversion in RX2's DC correction path

        This replicates the effect of 'rx1-rx2-phase-inversion-enable' on PlutoSDR.
        """

        # Shortcuts to libbladeRF C API functions
        get_register = bladerf._bladerf.libbladeRF.bladerf_get_rfic_register
        set_register = bladerf._bladerf.libbladeRF.bladerf_set_rfic_register

        # 'self.sdr' presumably is a bladerf.BladeRF() instance or similar
        # 'dev = self.sdr.dev[0]' references the raw C device handle array
        dev = self.sdr.dev[0]

        # -- Register 0x022: AGC Attack Delay & LNA polarity --
        val22_ptr = bladerf._bladerf.ffi.new("uint8_t *")
        # Read the current register value into 'val22_ptr[0]'
        get_register(dev, 0x22, val22_ptr)
        current_022 = val22_ptr[0]

        # Set bit 6 (0x40) to invert the signal phase when LNA is bypassed
        new_022 = current_022 | (1 << 6)  # (1 << 6) = 0x40
        set_register(dev, 0x22, new_022)
        print(f"Register 0x022: 0x{current_022:02X} -> 0x{new_022:02X}")

        # -- Register 0x011: Parallel Port Conf 2 --
        #    (Handles RX1/RX2 phase inversion)
        val11_ptr = bladerf._bladerf.ffi.new("uint8_t *")
        get_register(dev, 0x11, val11_ptr)
        current_011 = val11_ptr[0]

        # Set bit 5 (0x20) to enable phase inversion on RX2
        new_011 = current_011 | (1 << 5)
        set_register(dev, 0x11, new_011)
        print(f"Register 0x011: 0x{current_011:02X} -> 0x{new_011:02X}")

        # -- Register 0x0189: Invert Bits (RX DC correction paths) --
        val189_ptr = bladerf._bladerf.ffi.new("uint8_t *")
        get_register(dev, 0x189, val189_ptr)
        current_0189 = val189_ptr[0]

        # Clear bit 5 to ensure RX2 DC offset correction is NOT inverted now
        new_0189 = current_0189 & ~(1 << 5)
        set_register(dev, 0x189, new_0189)
        print(f"Register 0x0189: 0x{current_0189:02X} -> 0x{new_0189:02X}")

        # Check for missing parenthesis
        # Original code was missing a right parenthesis here.
        # Correct line:
        # print(f"Register 0x0189: 0x{current_0189:02X} -> 0x{new_0189:02X}")

        # (Optional) Verify by reading back the registers
        get_register(dev, 0x22, val22_ptr)
        get_register(dev, 0x11, val11_ptr)
        get_register(dev, 0x189, val189_ptr)

        print(
            f"New 0x022 = 0x{val22_ptr[0]:02X}, "
            + f"New 0x011 = 0x{val11_ptr[0]:02X}, "
            + f"New 0x189 = 0x{val189_ptr[0]:02X}"
        )

        for rx_ch_idx, rx_ch in enumerate(self.rx_channels):
            rx_ch.bandwidth = self.rx_config.rf_bandwidth
            assert rx_ch.bandwidth == self.rx_config.rf_bandwidth

            rx_ch.sample_rate = self.rx_config.sample_rate
            assert rx_ch.sample_rate == self.rx_config.sample_rate

            rx_ch.frequency = self.rx_config.lo
            assert (
                abs(rx_ch.frequency - self.rx_config.lo) < 10
            ), f"failed to set radio lo {rx_ch.frequency} != {self.rx_config.lo}"

            if self.rx_config.gain_control_modes[rx_ch_idx] == "slow_attack":
                rx_ch.gain_mode = (
                    bladerf._bladerf.GainMode.SlowAttack_AGC
                )  # FastAttack_AGC , Manual, Hybrid_AGC
            elif self.rx_config.gain_control_modes[rx_ch_idx] == "fast_attack":
                rx_ch.gain_mode = (
                    bladerf._bladerf.GainMode.FastAttack_AGC
                )  # FastAttack_AGC , Manual, Hybrid_AGC
            elif self.rx_config.gain_control_modes[rx_ch_idx] == "manual":
                rx_ch.gain_mode = (
                    bladerf._bladerf.GainMode.Manual
                )  # FastAttack_AGC , Manual, Hybrid_AGC
                rx_ch.gain = self.rx_config.gains[rx_ch_idx]
            else:
                assert 1 == 0
            rx_ch.enable = True

        # typedef enum {
        #     BLADERF_RX_X1 = 0, /**< x1 RX (SISO) */
        #     BLADERF_TX_X1 = 1, /**< x1 TX (SISO) */
        #     BLADERF_RX_X2 = 2, /**< x2 RX (MIMO) */
        #     BLADERF_TX_X2 = 3, /**< x2 TX (MIMO) */
        # } bladerf_channel_layout;
        #  * When a multi-channel ::bladerf_channel_layout is selected, samples
        # * will be interleaved per channel. For example, with ::BLADERF_RX_X2
        # * or ::BLADERF_TX_X2 (x2 MIMO), the buffer is structured like:
        # *
        # * <pre>
        # *  .-------------.--------------.--------------.------------------.
        # *  | Byte offset | Bits 31...16 | Bits 15...0  |    Description   |
        # *  +-------------+--------------+--------------+------------------+
        # *  |    0x00     |     Q0[0]    |     I0[0]    |  Ch 0, sample 0  |
        # *  |    0x04     |     Q1[0]    |     I1[0]    |  Ch 1, sample 0  |
        # *  |    0x08     |     Q0[1]    |     I0[1]    |  Ch 0, sample 1  |
        # *  |    0x0c     |     Q1[1]    |     I1[1]    |  Ch 1, sample 1  |
        # *  |    ...      |      ...     |      ...     |        ...       |
        # *  |    0xxx     |     Q0[n]    |     I0[n]    |  Ch 0, sample n  |
        # *  |    0xxx     |     Q1[n]    |     I1[n]    |  Ch 1, sample n  |
        # *  `-------------`--------------`--------------`------------------`
        # * </pre>
        self.sdr.sync_config(
            layout=bladerf._bladerf.ChannelLayout.RX_X2,
            fmt=bladerf._bladerf.Format.SC16_Q11,
            num_buffers=self.rx_config.rx_buffers,
            buffer_size=self.rx_config.buffer_size,  # size in samples
            num_transfers=1,
            stream_timeout=3500,
        )

    def setup_tx_config(self):
        pass

    def close_rx(self):
        pass

    def close_tx(self):
        pass

    def check_for_freq_peak(self):
        return True

    def rx(self):
        bytes_per_sample = 4  # 4 bytes, 16bit I, 16bit Q
        buf = bytearray(self.rx_config.buffer_size * bytes_per_sample * 2)  # 2 channels
        self.sdr.sync_rx(buf, self.rx_config.buffer_size * 2)
        samples = np.frombuffer(buf, dtype=np.int16)
        samples = samples[0::2] + 1j * samples[1::2]  # Convert to complex type
        # samples /= 2048.0  # Scale to -1 to 1 (its using 12 bit ADC)
        samples = np.vstack([samples[::2], samples[1::2]])
        return samples


class FakePPlus:
    def __init__(
        self,
        uri: str,
        rx_config: ReceiverConfig = None,
        tx_config: EmitterConfig = None,
        phase_calibration=0.0,
    ):
        super(FakePPlus, self).__init__()
        self.uri = uri

        self.tx_config = None
        self.rx_config = None
        self.set_config(rx_config=rx_config, tx_config=tx_config)
        self.sdr = FakeSdr()

    def rx(self, num_samples=None):
        return self.sdr.rx()

    def soft_reset_radio(self):
        time.sleep(0.1)

    def rssis(self):
        return np.random.rand(2)

    def gains(self):
        return np.random.rand(2)

    def get_rssi_and_gain(self):
        """
        The original code used ADI debug attrs to read RSSI, hardware gains, etc.
        bladeRF does not provide direct 'RSSI' readouts in the same way.

        You *could* approximate by reading RSSI from actual samples (e.g. measure power).
        Below is a placeholder.
        """
        # Pseudo-code: measure a short burst, compute power -> 'RSSI'
        # Gains are typically set on LNA, VGA1, VGA2 for bladeRF.
        # We'll just provide some placeholders for demonstration.

        # For example, read the current front-end gains:
        lna_gain = self.dev.lna_gain  # e.g. "max" or a numeric
        rxvga1 = self.dev.vga1  # typically an integer in dB
        rxvga2 = self.dev.vga2  # also dB

        # Placeholder 'RSSI' values
        measured_rssi_0 = -50.0
        measured_rssi_1 = -50.0

        # Return an array of [RSSI0, RSSI1, gain0, gain1]
        return np.array(
            [
                measured_rssi_0,
                measured_rssi_1,
                float(rxvga1),
                float(rxvga2),
            ]
        )

    def set_config(
        self, rx_config: ReceiverConfig = None, tx_config: EmitterConfig = None
    ):
        logging.debug(f"{self.uri}: set_config RX{str(rx_config)} TX{str(tx_config)})")
        # RX should be setup like this
        if rx_config is not None:
            assert self.rx_config is None
            self.rx_config = rx_config
            self.sdr = FakeSdr()

        # TX should be setup like this
        if tx_config is not None:
            assert self.tx_config is None
            self.tx_config = tx_config

    def close(self):
        pass

    def setup_rx_config(self):
        self.sdr.set_buffer_size(self.rx_config.buffer_size)

    def setup_tx_config(self):
        pass

    def close_rx(self):
        pass

    def close_tx(self):
        pass

    def check_for_freq_peak(self):
        return True


class PPlus:
    def __init__(
        self,
        uri: str,
        rx_config: ReceiverConfig = None,
        tx_config: EmitterConfig = None,
        phase_calibration=0.0,
    ):
        super(PPlus, self).__init__()
        self.uri = get_uri(rx_config=rx_config, tx_config=tx_config, uri=uri)
        logging.info(f"{self.uri}: Open PlutoPlus")

        # try to fix issue with radios coming online
        self.sdr = adi.ad9361(uri=self.uri)
        self.close_tx()
        # self.sdr = None
        time.sleep(0.5)

        # open for real
        # self.sdr = adi.ad9361(uri=self.uri)
        self.tx_config = None
        self.rx_config = None
        self.set_config(rx_config=rx_config, tx_config=tx_config)
        self.phase_calibration = phase_calibration

        if (
            self.tx_config is None and self.rx_config is None
        ):  # this is a fresh open or reset
            self.close_tx()
            self.sdr.tx_destroy_buffer()
            self.sdr.rx_destroy_buffer()
            self.sdr.tx_enabled_channels = []

    def rx(self, num_samples=None):
        return self.sdr.rx()

    def soft_reset_radio(self):
        return  # disable reset
        old_freq = self.sdr.rx_lo
        self.sdr.rx_lo = 700_000_000  # 700Mhz
        time.sleep(0.05)
        self.sdr.rx_lo = old_freq
        time.sleep(0.05)

    def get_rssi_and_gain(self):
        v0 = self.sdr._ctrl.find_channel("voltage0")
        v1 = self.sdr._ctrl.find_channel("voltage1")
        # breakpoint()
        return np.array(
            [
                float(v0.attrs["rssi"].value[:-3]),
                float(v1.attrs["rssi"].value[:-3]),
                float(v0.attrs["hardwaregain"].value[:-3]),
                float(v1.attrs["hardwaregain"].value[:-3]),
                # self.sdr._get_iio_attr("voltage0", "hardwaregain", False),
                # self.sdr._get_iio_attr("voltage1", "hardwaregain", False),
            ]
        )

    def rssis(self):
        return np.array(
            [
                float(self.sdr._ctrl.find_channel("voltage0").attrs["rssi"].value[:-3]),
                float(self.sdr._ctrl.find_channel("voltage1").attrs["rssi"].value[:-3]),
            ]
        )

    def gains(self):
        return np.array(
            [
                self.sdr._get_iio_attr("voltage0", "hardwaregain", False),
                self.sdr._get_iio_attr("voltage1", "hardwaregain", False),
            ]
        )

    def set_config(
        self, rx_config: ReceiverConfig = None, tx_config: EmitterConfig = None
    ):
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
        # logging.debug(f"{self.uri}: Start delete PlutoPlus")
        self.close_tx()
        # self.sdr.tx_destroy_buffer()
        # self.sdr.rx_destroy_buffer()
        self.sdr.tx_enabled_channels = []
        # logging.debug(f"{self.uri}: Done delete PlutoPlus")

    """
    Setup the Rx part of the pluto
    """

    def setup_rx_config(self):
        # disable the channels before changing
        # self.sdr.rx_enabled_channels = []
        # assert len(self.sdr.rx_enabled_channels) == 0
        self.sdr.rx_destroy_buffer()

        # Fix the phase inversion on channel RX1
        self.sdr._ctrl.debug_attrs["adi,rx1-rx2-phase-inversion-enable"].value = "1"
        assert (
            self.sdr._ctrl.debug_attrs["adi,rx1-rx2-phase-inversion-enable"].value
            == "1"
        )

        # https://www.analog.com/media/cn/technical-documentation/user-guides/ad9364_register_map_reference_manual_ug-672.pdf
        reg22_value = self.sdr._ctrl.reg_read(0x22)
        # https://github.com/analogdevicesinc/linux/blob/88946d52e61d5b898c061d820caf27fd1c3730d7/drivers/iio/adc/ad9361_regs.h#L780
        self.sdr._ctrl.reg_write(0x22, reg22_value | (1 << 6))
        reg22_value = self.sdr._ctrl.reg_read(0x22)
        assert (reg22_value & (1 << 6)) != 0

        # self.sdr._ctrl.reg_write(0x22, reg22_value & (~(1 << 6)))
        # reg22_value = self.sdr._ctrl.reg_read(0x22)
        # assert (reg22_value & (1 << 6)) == 0

        self.sdr.rx_rf_bandwidth = self.rx_config.rf_bandwidth
        assert self.sdr.rx_rf_bandwidth == self.rx_config.rf_bandwidth

        # some of these reset the controller and gain settings when set, so always set them!

        # if self.sdr.sample_rate != self.rx_config.sample_rate:
        self.sdr.sample_rate = self.rx_config.sample_rate
        assert self.sdr.sample_rate == self.rx_config.sample_rate

        self.sdr.rx_lo = self.rx_config.lo
        assert (
            abs(self.sdr.rx_lo - self.rx_config.lo) < 10
        ), f"failed to set radio lo {self.sdr.rx_lo} != {self.rx_config.lo}"

        # set filter_fir_en according to config
        for rx_channel in [
            self.sdr._ctrl.find_channel("voltage0", is_output=False),
            self.sdr._ctrl.find_channel("voltage1", is_output=False),
        ]:
            if rx_channel.attrs["filter_fir_en"].value != str(
                self.rx_config.filter_fir_en
            ):
                rx_channel.attrs["filter_fir_en"].value = str(
                    self.rx_config.filter_fir_en
                )
            assert rx_channel.attrs["filter_fir_en"].value == str(
                self.rx_config.filter_fir_en
            )

        # setup the gain mode
        self.sdr.gain_control_mode_chan0 = self.rx_config.gain_control_modes[0]
        assert self.sdr.gain_control_mode_chan0 == self.rx_config.gain_control_modes[0]
        if self.rx_config.gain_control_modes[0] == "manual":
            self.sdr.rx_hardwaregain_chan0 = self.rx_config.gains[0]
            assert self.sdr.rx_hardwaregain_chan0 == self.rx_config.gains[0]

        self.sdr.gain_control_mode_chan1 = self.rx_config.gain_control_modes[1]
        assert self.sdr.gain_control_mode_chan1 == self.rx_config.gain_control_modes[1]
        if self.rx_config.gain_control_modes[1] == "manual":
            self.sdr.rx_hardwaregain_chan1 = self.rx_config.gains[1]
            assert self.sdr.rx_hardwaregain_chan1 == self.rx_config.gains[1]

        if self.rx_config.buffer_size is not None:
            self.sdr.rx_buffer_size = self.rx_config.buffer_size

        self.sdr._rxadc.set_kernel_buffers_count(
            self.rx_config.rx_buffers
        )  # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto

        self.sdr.rx_enabled_channels = self.rx_config.enabled_channels

        # turn off TX
        # TODO this could cause an issue when actually using TX
        self.sdr._ctrl.find_channel("altvoltage1", is_output=True).attrs[
            "powerdown"
        ].value = "1"
        # default setting for RX
        if (
            self.sdr._ctrl.find_channel("altvoltage0", is_output=True).attrs[
                "powerdown"
            ]
            != "0"
        ):
            self.sdr._ctrl.find_channel("altvoltage0", is_output=True).attrs[
                "powerdown"
            ].value = "0"

    """
    Setup the Tx side of the pluto
    """

    def setup_tx_config(self):
        logging.debug(f"{self.tx_config.uri}: Setup TX")

        self.sdr.tx_destroy_buffer()
        # self.sdr.tx_enabled_channels = []
        # assert len(self.sdr.tx_enabled_channels) == 0

        self.sdr.tx_cyclic_buffer = self.tx_config.cyclic  # this keeps repeating!
        assert self.sdr.tx_cyclic_buffer == self.tx_config.cyclic

        # some of these reset the controller and gain settings when set, so always set them!

        self.sdr.tx_rf_bandwidth = self.tx_config.rf_bandwidth
        assert self.sdr.tx_rf_bandwidth == self.tx_config.rf_bandwidth

        self.sdr.sample_rate = self.tx_config.sample_rate
        assert self.sdr.sample_rate == self.tx_config.sample_rate

        self.sdr.tx_lo = self.tx_config.lo
        assert self.sdr.tx_lo == self.tx_config.lo

        # setup the gain mode
        self.sdr.tx_hardwaregain_chan0 = self.tx_config.gains[0]
        assert self.sdr.tx_hardwaregain_chan0 == self.tx_config.gains[0]
        self.sdr.tx_hardwaregain_chan1 = self.tx_config.gains[1]
        assert self.sdr.tx_hardwaregain_chan1 == self.tx_config.gains[1]

        if self.tx_config.buffer_size is not None:
            self.sdr.tx_buffer_size = self.tx_config.buffer_size
        # self.sdr._rxadc.set_kernel_buffers_count(
        #    1
        # )  # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
        self.sdr.tx_enabled_channels = self.tx_config.enabled_channels
        time.sleep(1)

    """
    Close pluto
    """

    def close_tx(self):
        try:
            self.sdr.tx_hardwaregain_chan0 = -80
            self.sdr.tx_hardwaregain_chan1 = -80
            self.sdr.tx_enabled_channels = []
            self.sdr.tx_destroy_buffer()
            self.sdr.tx_cyclic_buffer = False
        except TypeError:
            pass
        self.tx_config = None
        # time.sleep(1.0)

    def close_rx(self):
        self.rx_config = None

    """
    Given an online SDR receiver check if the max power peak is as expected during calibration
    """

    def check_for_freq_peak(self):
        freq = np.fft.fftfreq(
            self.rx_config.buffer_size, d=1.0 / self.rx_config.sample_rate
        )
        signal_matrix = np.vstack(self.rx())
        sp = np.fft.fft(signal_matrix[0])
        max_freq = freq[np.abs(np.argmax(sp.real))]
        if np.abs(max_freq - self.rx_config.intermediate) < (
            self.rx_config.sample_rate / self.rx_config.buffer_size + 1
        ):
            return True
        return False


def make_tone(tx_config: EmitterConfig):
    # create a buffe for the signal
    fc0 = tx_config.intermediate
    fs = tx_config.sample_rate  # must be <=30.72 MHz if both channels are enabled
    tx_n = int(fs / gcd(fs, fc0))
    while tx_n < 1024 * 2:
        tx_n *= 2

    # since its a cyclic buffer its important to end on a full phase
    t = (
        np.arange(0, tx_n) / fs
    )  # time at each point assuming we are sending samples at (1/fs)s
    return np.exp(1j * 2 * np.pi * fc0 * t) * (2**14)


def setup_rx(rx_config, provided_pplus_rx=None):
    logging.info(f"setup_rx({rx_config.uri}) skip retries")
    # sdr_rx = adi.ad9361(uri=receiver_uri)
    if provided_pplus_rx is None:
        logging.debug(f"{rx_config.uri} RX using external tx")
        pplus_rx = get_pplus(rx_config=rx_config)
        pplus_rx.setup_rx_config()
    else:
        # TODO if pluto_rx is provided confirm its the same config
        pplus_rx = provided_pplus_rx
    time.sleep(2)
    # get RX and drop it
    for _ in range(10):
        pplus_rx.rx()

    logging.info(f"RX came online with config\nRX_config:{pplus_rx.rx_config}")
    pplus_rx.phase_calibration = 0.0
    return pplus_rx


def setup_rxtx(rx_config, tx_config, leave_tx_on=False, provided_pplus_rx=None):
    retries = 0
    if provided_pplus_rx is None:
        pplus_rx = get_pplus(rx_config=rx_config)
        pplus_rx.setup_rx_config()
    else:
        pplus_rx = provided_pplus_rx
    time.sleep(1)
    # get RX and drop it
    for _ in range(400):
        pplus_rx.rx()

    while run_radios and retries < 15:
        logging.info(f"setup_rxtx({rx_config.uri}, {tx_config.uri}) retry {retries}")
        # sdr_rx = adi.ad9361(uri=receiver_uri)
        pplus_tx = get_pplus(tx_config=tx_config)

        pplus_tx.setup_tx_config()
        time.sleep(0.5)

        # start TX
        pplus_tx.sdr.tx(make_tone(tx_config))
        time.sleep(2.0)

        # get RX and drop it
        for _ in range(400):
            pplus_rx.rx()

        # test to see what frequency we are seeing
        if pplus_rx.check_for_freq_peak():
            logging.info(
                f"RXTX came online with configs\nRX_config:{pplus_rx.rx_config}\nTX_config:{pplus_tx.tx_config}"
            )
            if not leave_tx_on:
                pplus_tx.close_tx()
            pplus_rx.phase_calibration = 0.0
            return pplus_rx, pplus_tx
        # if provided_pplus_rx is None:
        #    pplus_rx.close_rx()
        pplus_tx.close_tx()
        retries += 1

        # try to reset
        # pplus_tx.close_tx()
        time.sleep(1)
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

    # make sure no other emitters online
    pplus_rx = get_pplus(rx_config=rx_config)
    pplus_rx.setup_rx()
    if pplus_rx.check_for_freq_peak():
        logging.error("Refusing phase calibration when another emitter online!")
        return None, None
    pplus_rx.close_rx()
    time.sleep(0.1)

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
            signal_matrix = np.vstack(pplus_rx.rx())
            phase_calibrations[idx] = (
                pi_norm(np.angle(signal_matrix[0]) - np.angle(signal_matrix[1]))
            ).mean()  # TODO THIS BREAKS if diff is near 2*np.pi...
            phase_calibrations_cm[idx], _ = circular_mean(
                (np.angle(signal_matrix[0]) - np.angle(signal_matrix[1])).reshape(
                    1, -1
                ),
                20,
            )
        phase_calibration_u = phase_calibrations.mean()
        phase_calibration_std = phase_calibrations.std()
        logging.info(
            f"{rx_config.uri}: Phase calibration mean \
                ({phase_calibration_u:0.4f}) std ({phase_calibration_std:0.4f})"
        )

        # TODO this part should also get replaced by circular mean
        phase_calibration_cm_u = circular_mean(
            phase_calibrations_cm.reshape(1, -1), 20
        )[0]
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
            pplus_rx.phase_calibration = circular_mean(
                phase_calibrations_cm.reshape(1, -1), 20
            )[
                0
            ]  # .mean()
            return pplus_rx, pplus_tx
    pplus_tx.close()
    logging.error(f"{rx_config.uri}: Phase calibration failed")
    return None, None


def plot_recv_signal(
    pplus_rx, nthetas, fig=None, axs=None, frames=-1, title=None, signal_matrixs=None
):
    if fig is None:
        fig, axs = plt.subplots(2, 4, figsize=(16, 6), layout="constrained")

    rx_n = pplus_rx.rx_config.buffer_size
    t = np.arange(rx_n)
    frame_idx = 0

    steering_vectors = precompute_steering_vectors(
        receiver_positions=pplus_rx.rx_config.rx_pos,
        carrier_frequency=pplus_rx.rx_config.lo,
        spacing=nthetas,
    )

    while run_radios and frame_idx != frames:
        pplus_rx.soft_reset_radio()
        if signal_matrixs is None:
            signal_matrix = np.vstack(pplus_rx.rx())
            # signal_matrix[1] *= np.exp(1j * pplus_rx.phase_calibration)
        else:
            signal_matrix = signal_matrixs[frame_idx]
        assert pplus_rx.rx_config.rx_pos is not None

        slow_beamformer = False
        if slow_beamformer:
            beam_thetas, beam_sds, _ = beamformer(
                pplus_rx.rx_config.rx_pos,
                signal_matrix,
                pplus_rx.rx_config.lo,
                spacing=nthetas,
            )
        else:
            # signal_matrix = np.vstack(signal_matrix)
            beam_thetas = beamformer_thetas(nthetas)
            beam_sds = beamformer_given_steering(
                steering_vectors=steering_vectors, signal_matrix=signal_matrix
            )

        freq = np.fft.fftfreq(t.shape[-1], d=1.0 / pplus_rx.rx_config.sample_rate)
        assert t.shape[-1] == rx_n
        for idx in [0, 1]:
            axs[idx][0].clear()
            axs[idx][1].clear()

            axs[idx][0].scatter(t, signal_matrix[idx].real, s=1, alpha=0.1)
            axs[idx][0].set_xlabel("Time")
            axs[idx][0].set_ylabel("Real(signal)")
            axs[idx][0].set_ylim([-1000, 1000])

            sp = np.fft.fft(signal_matrix[idx])
            axs[idx][1].scatter(
                freq, np.log(np.abs(sp.real)), s=1, alpha=0.1
            )  # , freq, sp.imag)
            axs[idx][1].set_xlabel("Frequency bin")
            axs[idx][1].set_ylabel("Power")
            axs[idx][1].set_ylim([-30, 30])
            max_freq = freq[np.abs(np.argmax(sp.real))]
            axs[idx][1].axvline(x=max_freq, label="max %0.2e" % max_freq, color="red")
            axs[idx][1].legend()

            axs[idx][2].clear()
            axs[idx][2].scatter(
                signal_matrix[idx].real, signal_matrix[idx].imag, s=1, alpha=0.1
            )
            axs[idx][2].set_xlabel("I real(signal)")
            axs[idx][2].set_ylabel("Q imag(signal)")
            axs[idx][2].set_title("IQ plot recv (%d)" % idx)
            axs[idx][2].set_ylim([-600, 600])
            axs[idx][2].set_xlim([-600, 600])

            axs[idx][0].set_title("Real signal recv (%d)" % idx)
            axs[idx][1].set_title("Power recv (%d)" % idx)
        diff = pi_norm(np.angle(signal_matrix[0]) - np.angle(signal_matrix[1]))
        axs[0][3].clear()
        axs[0][3].scatter(t, diff, s=0.3, alpha=0.1)
        mean, _mean = circular_mean(diff.reshape(1, -1), 20)
        axs[0][3].axhline(y=mean, color="black", label="circular mean")
        axs[0][3].axhline(y=_mean, color="red", label="trimmed circular mean")
        axs[0][3].set_ylim([-np.pi, np.pi])
        axs[0][3].set_xlabel("Time")
        axs[0][3].set_ylabel("Angle estimate")
        axs[0][3].legend()
        axs[0][3].set_title("Phase_RX0 - Phase_RX1")
        axs[1][3].clear()
        axs[1][3].plot(beam_thetas, beam_sds)

        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{pplus_rx.uri} phase cal: {pplus_rx.phase_calibration:0.4f}")

        # plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.00001)
        frame_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receiver-uri",
        type=str,
        help="Receivers",
        required=True,
    )
    parser.add_argument(
        "--emitter-uri", type=str, help="Long term emitter", required=False
    )
    parser.add_argument("-l", "--leave-tx-on", action=argparse.BooleanOptionalAction)
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
        "--nthetas",
        type=int,
        help="thetas",
        required=False,
        default=int(64 + 1),
    )
    parser.add_argument(
        "--fs",
        type=int,
        help="Sampling frequency",
        required=False,
        default=int(16e6),
    )
    # parser.add_argument(
    #    "--cal0",
    #    type=int,
    #    help="Rx0 calibration phase offset in degrees",
    #    required=False,
    #    default=180,
    # )
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
        default="fast_attack",
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
        "--kernel-buffers",
        type=int,
        help="kernel buffers",
        required=False,
        default=2,
    )  # 12
    parser.add_argument(
        "--rx-spacing",
        type=float,
        help="RX spacing",
        required=False,
        default=0.065,
    )  # 12
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction)
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
        level="INFO",
    )

    if args.mode == "rxcal":
        # if we use weaker tx gain then the noise in phase calibration goes up
        tx_config = args_to_tx_config(args)
        # tx_config.gains = [-30, -80]
        pplus_rx, pplus_tx = setup_rxtx_and_phase_calibration(
            rx_config=args_to_rx_config(args),
            tx_config=tx_config,
            leave_tx_on=args.leave_tx_on,
        )
        # pplus_tx.sdr.tx_hardwaregain_chan0=tx_config

        if pplus_rx is None:
            logging.error("Failed phase calibration, exiting")
            sys.exit(1)

        plot_recv_signal(pplus_rx, args.nthetas)

    elif args.mode == "rx":
        # pplus_rx = get_pplus(rx_config=args_to_rx_config(args))
        # pplus_rx.setup_rx_config()
        pplus_rx = setup_rx(rx_config=args_to_rx_config(args))
        # pplus_rx.phase_calibration = args.cal0
        if args.benchmark:
            steering_vectors = precompute_steering_vectors(
                receiver_positions=pplus_rx.rx_config.rx_pos,
                carrier_frequency=pplus_rx.rx_config.lo,
                spacing=args.nthetas,
            )
            for _ in tqdm(range(int(1e6))):
                signal_matrix = np.vstack(pplus_rx.rx())
                rssis = pplus_rx.rssis()
                gains = pplus_rx.gains()

                beam_sds = beamformer_given_steering(
                    steering_vectors=steering_vectors, signal_matrix=signal_matrix
                )
        else:
            plot_recv_signal(pplus_rx, args.nthetas)
    elif args.mode == "tx":
        pplus_rx, pplus_tx = setup_rxtx(
            rx_config=args_to_rx_config(args),
            tx_config=args_to_tx_config(args),
            leave_tx_on=True,
        )
        if pplus_rx is None:
            logging.error("Failed to bring emitter online")
            sys.exit(1)
        logging.info(
            f"{args.emitter_uri}: Emitter online verified by {args.receiver_uri}"
        )
        # apply the previous calibration
        time.sleep(1800)

import argparse
import time

import adi  # Analog Devices library for AD9361 and other devices
import numpy as np


class SDRBenchmark:
    def __init__(self, uri, rx_config):
        self.sdr = adi.ad9361(uri=uri)
        self.rx_config = rx_config

    def setup_rx_config(self):
        """Sets up the SDR with the provided RX configuration."""
        # Destroy existing buffers to reconfigure
        self.sdr.rx_destroy_buffer()

        # Fix phase inversion on RX1
        self.sdr._ctrl.debug_attrs["adi,rx1-rx2-phase-inversion-enable"].value = "1"
        assert (
            self.sdr._ctrl.debug_attrs["adi,rx1-rx2-phase-inversion-enable"].value
            == "1"
        )

        # Update register 0x22 to enable necessary configuration
        reg22_value = self.sdr._ctrl.reg_read(0x22)
        self.sdr._ctrl.reg_write(0x22, reg22_value | (1 << 6))
        reg22_value = self.sdr._ctrl.reg_read(0x22)
        assert (reg22_value & (1 << 6)) != 0

        # Set RX bandwidth
        self.sdr.rx_rf_bandwidth = self.rx_config["rf_bandwidth"]
        assert self.sdr.rx_rf_bandwidth == self.rx_config["rf_bandwidth"]

        # Set sample rate
        self.sdr.sample_rate = self.rx_config["sample_rate"]
        assert self.sdr.sample_rate == self.rx_config["sample_rate"]

        # Set RX LO
        self.sdr.rx_lo = self.rx_config["lo"]
        assert (
            abs(self.sdr.rx_lo - self.rx_config["lo"]) < 10
        ), f"Failed to set LO: {self.sdr.rx_lo} != {self.rx_config['lo']}"

        # Set gain control mode and manual gain (if applicable)
        self.sdr.gain_control_mode_chan0 = self.rx_config["gain_control_modes"][0]
        assert (
            self.sdr.gain_control_mode_chan0 == self.rx_config["gain_control_modes"][0]
        )
        if self.rx_config["gain_control_modes"][0] == "manual":
            self.sdr.rx_hardwaregain_chan0 = self.rx_config["gains"][0]
            assert self.sdr.rx_hardwaregain_chan0 == self.rx_config["gains"][0]

        self.sdr.gain_control_mode_chan1 = self.rx_config["gain_control_modes"][1]
        assert (
            self.sdr.gain_control_mode_chan1 == self.rx_config["gain_control_modes"][1]
        )
        if self.rx_config["gain_control_modes"][1] == "manual":
            self.sdr.rx_hardwaregain_chan1 = self.rx_config["gains"][1]
            assert self.sdr.rx_hardwaregain_chan1 == self.rx_config["gains"][1]

        # Set RX buffer size
        if self.rx_config["buffer_size"] is not None:
            self.sdr.rx_buffer_size = self.rx_config["buffer_size"]

        # Set kernel buffers count
        self.sdr._rxadc.set_kernel_buffers_count(self.rx_config["rx_buffers"])

        # Enable RX channels
        self.sdr.rx_enabled_channels = self.rx_config["enabled_channels"]

    def benchmark_sample_rate(self, num_buffers=100):
        """Measures the effective sample rate by streaming data."""
        start_time = time.time()
        total_samples = 0

        for _ in range(num_buffers):
            samples = self.sdr.rx()
            total_samples += len(samples[0]) * 2  # Assuming complex I/Q data

        elapsed_time = time.time() - start_time
        effective_sample_rate = total_samples / elapsed_time
        return effective_sample_rate


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Benchmark sample rate for AD9361.")
    parser.add_argument(
        "--uri",
        type=str,
        required=True,
        help="URI for the SDR (e.g., usb:, ip:192.168.2.1).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000000,
        help="Sample rate in Sps (default: 16 MSPS).",
    )
    parser.add_argument(
        "--lo",
        type=int,
        default=2412000000,
        help="RX LO frequency in Hz (default: 2.412 GHz for Wi-Fi Channel 1).",
    )
    parser.add_argument(
        "--rf_bandwidth",
        type=int,
        default=20000000,
        help="RF bandwidth in Hz (default: 20 MHz).",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=500000,
        help="RX buffer size (default: 500,000 samples).",
    )
    parser.add_argument(
        "--rx_buffers",
        type=int,
        default=1,
        help="Number of kernel buffers (default: 1).",
    )
    parser.add_argument(
        "--gain_control_modes",
        nargs=2,
        default=["slow_attack", "slow_attack"],
        help="Gain control modes for channels 0 and 1 (default: ['slow_attack', 'slow_attack']).",
    )
    parser.add_argument(
        "--gains",
        nargs=2,
        type=int,
        default=[40, 40],
        help="Manual gain settings (default: 40 dB).",
    )
    parser.add_argument(
        "--enabled_channels",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Enabled RX channels (default: [0, 1]).",
    )
    parser.add_argument(
        "--num_buffers",
        type=int,
        default=100,
        help="Number of buffers for benchmarking.",
    )

    args = parser.parse_args()

    # RX configuration dictionary
    rx_config = {
        "sample_rate": args.sample_rate,
        "lo": args.lo,
        "rf_bandwidth": args.rf_bandwidth,
        "buffer_size": args.buffer_size,
        "rx_buffers": args.rx_buffers,
        "gain_control_modes": args.gain_control_modes,
        "gains": args.gains,
        "enabled_channels": args.enabled_channels,
    }

    # Initialize and configure SDR
    sdr_benchmark = SDRBenchmark(uri=args.uri, rx_config=rx_config)
    sdr_benchmark.setup_rx_config()

    # Benchmark sample rate
    effective_sample_rate = sdr_benchmark.benchmark_sample_rate(
        num_buffers=args.num_buffers
    )
    print(f"Effective Sample Rate: {effective_sample_rate / 1e6:.2f} MSPS")


if __name__ == "__main__":
    main()

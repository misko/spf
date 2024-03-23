import argparse
import time

import adi
from tqdm import tqdm


def get_sdr(uri, rx_buffer_size=2 * 16):
    sdr = adi.ad9361(uri=uri)
    # Create radio
    # Configure properties
    sdr.sample_rate = 16000000
    sdr.tx_lo = 2412000000
    sdr.tx_cyclic_buffer = True
    sdr.tx_rf_bandwidth = 3 * 100000
    sdr.tx_hardwaregain_chan0 = -5
    sdr.rx_buffer_size = rx_buffer_size
    sdr.gain_control_mode_chan0 = "fast_attack"
    sdr.gain_control_mode_chan1 = "fast_attack"
    sdr.tx_enabled_channels = [0]
    return sdr


def benchmark(uri, rx_buffer_size, total_samples=2**24):
    sdr = get_sdr(uri, rx_buffer_size=rx_buffer_size)
    assert total_samples % rx_buffer_size == 0
    assert sdr.rx()[0].shape[0] == rx_buffer_size
    start_time = time.time()

    for _ in tqdm(range(total_samples // rx_buffer_size)):
        sdr.rx()
    elapsed_time = time.time() - start_time
    total_bits = total_samples * 2 * 128
    return {
        "total_time": elapsed_time,
        "samples_per_second": total_samples / elapsed_time,
        "bits_per_second": total_bits / elapsed_time,
        "Mbits_per_second": total_bits / (elapsed_time * 1024 * 1024),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        type=str,
        help="URI of radio",
        required=True,
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        help="buffer size",
        required=True,
    )
    args = parser.parse_args()

    for k, v in benchmark(args.uri, args.buffer_size).items():
        print(f"{k}: {v}")

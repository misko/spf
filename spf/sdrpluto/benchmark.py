import argparse
import threading
import time
from queue import Queue

import adi
import numpy as np
import zarr
import zarr.storage
from numcodecs import Blosc, Zstd
from tqdm import tqdm


def get_sdr(uri, rx_buffer_size=2 * 16, rx_buffers=None):
    sdr = adi.ad9361(uri=uri)
    # Create radio
    # Configure properties
    sdr.sample_rate = 30000000  # 16000000
    sdr.tx_lo = 2412000000
    sdr.tx_cyclic_buffer = True
    sdr.tx_rf_bandwidth = 3 * 100000
    sdr.tx_hardwaregain_chan0 = -5
    sdr.rx_buffer_size = rx_buffer_size
    sdr.gain_control_mode_chan0 = "fast_attack"
    sdr.gain_control_mode_chan1 = "fast_attack"
    sdr.tx_enabled_channels = [0]
    if rx_buffers is not None:
        sdr._rxadc.set_kernel_buffers_count(rx_buffers)

    return sdr


def benchmark(
    uri,
    rx_buffer_size,
    total_samples=2**24,
    rx_buffers=None,
    filename=None,
    compress=None,
    chunk_size=1024,
    store=None,
):

    z = None
    if filename is not None:
        compressor = None
        if compress is not None:
            if "zstd" in compress:
                compressor = Zstd(level=int(compress.split("zstd")[1]))
            elif "none" == compress:
                compressor = None
            elif "blosc" in compress:
                compressor = Blosc(
                    cname="zstd",
                    clevel=int(compress.split("blosc")[1]),
                    shuffle=Blosc.BITSHUFFLE,
                )
            else:
                raise NotImplementedError
        if store is None or store == "directory":
            store = zarr.DirectoryStore(filename)
        elif store == "lmdb":
            store = zarr.LMDBStore(
                filename, map_size=2**38, writemap=True, map_async=True
            )
        else:
            raise NotImplementedError
        z = zarr.open(
            store=store,
            mode="w",
            shape=(2, total_samples),
            chunks=(1, 1024 * chunk_size),
            dtype="complex128",
            compressor=compressor,
        )
    sdr = get_sdr(uri, rx_buffer_size=rx_buffer_size, rx_buffers=rx_buffers)
    assert total_samples % rx_buffer_size == 0
    assert sdr.rx()[0].shape[0] == rx_buffer_size

    queue = Queue(maxsize=2)

    # threading
    def write_to_disk():
        idx = 0
        while True:
            v = queue.get()
            if v is None:
                return
            z[:, idx * rx_buffer_size : (idx + 1) * rx_buffer_size] = np.vstack(v)
            idx += 1

    write_thread = threading.Thread(target=write_to_disk)
    write_thread.start()

    time.sleep(1.0)
    start_time = time.time()
    for idx in tqdm(range(total_samples // rx_buffer_size)):
        queue.put(sdr.rx())
    queue.put(None)

    write_thread.join()
    elapsed_time = time.time() - start_time
    total_bits = total_samples * 2 * 128
    return {
        "total_time": elapsed_time,
        "samples_per_second": total_samples / elapsed_time,
        "bits_per_second": total_bits / elapsed_time,
        "Mbits_per_second": total_bits / (elapsed_time * 1024 * 1024),
        "data_bytes": z.nbytes,
        "data_disk_bytes": z.nbytes_stored,
        "data_ratio": z.nbytes / (z.nbytes_stored + 1.0),
        "data": z,
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
        "--buffer-sizes",
        type=str,
        help="buffer size  (can use 2**X)",
        required=False,
        nargs="+",
        default="2**20",
    )
    parser.add_argument(
        "--total-samples",
        type=str,
        help="buffer size  (can use 2**X)",
        required=False,
        default="2**24",
    )
    parser.add_argument(
        "--rx-buffers", type=int, help="how many rx buffers", required=False, default=4
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        help="1024 chunk multipied",
        required=False,
        nargs="+",
        default=1024,
    )
    parser.add_argument(
        "--write-to-file", type=str, help="file", required=False, default=None
    )
    parser.add_argument(
        "--compress", type=str, help="file", required=False, default="none", nargs="+"
    )
    parser.add_argument(
        "--stores",
        type=str,
        help="store",
        required=False,
        default="directory",
        nargs="+",
    )

    args = parser.parse_args()

    def numify(x):
        if "2**" in x:
            return 2 ** (int(x.split("**")[1]))
        else:
            return int(x)

    buffer_sizes = [numify(buffer_size) for buffer_size in args.buffer_sizes]
    args.total_samples = numify(args.total_samples)
    print("compression\tstore\tbuffer_size\tchunk_size\tproperty\tvalue")
    for compress in args.compress:
        for store in args.stores:
            for buffer_size in buffer_sizes:
                for chunk_size in args.chunk_sizes:
                    for k, v in benchmark(
                        args.uri,
                        buffer_size,
                        store=store,
                        filename=args.write_to_file,
                        compress=compress,
                        chunk_size=chunk_size,
                    ).items():
                        print(
                            f"{compress}\t{store}\t{buffer_size}\t{chunk_size}\t{k}\t{v}"
                        )

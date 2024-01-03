import argparse

import adi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, help="ip", required=True)
    args = parser.parse_args()
    sdr = adi.ad9361(uri="ip:%s" % args.ip)
    sdr.tx_enabled_channels = []
    sdr.tx_hardwaregain_chan0 = -80
    sdr.tx_hardwaregain_chan1 = -80
    sdr.tx_destroy_buffer()

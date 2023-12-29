import argparse

import adi

from spf.sdrpluto.sdr_controller import close_tx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, help="ip", required=True)
    args = parser.parse_args()
    sdr = adi.ad9361(uri="ip:%s" % args.ip)
    close_tx(sdr)

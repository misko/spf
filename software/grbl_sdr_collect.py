from sdrpluto.gather import * 
from grbl.grbl_interactive import GRBLManager 
import threading
import time
import numpy as np

def bounce_grbl(gm):
    while True:
        gm.bounce(10)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-ip", type=str, help="receiver Pluto IP address",required=True)
    parser.add_argument("--emitter-ip", type=str, help="emitter Pluto IP address",required=True)
    parser.add_argument("--fi", type=int, help="Intermediate frequency",required=False,default=1e5)
    parser.add_argument("--fc", type=int, help="Carrier frequency",required=False,default=2.5e9)
    parser.add_argument("--fs", type=int, help="Sampling frequency",required=False,default=16e6)
    parser.add_argument("--cal0", type=int, help="Rx0 calibration phase offset in degrees",required=False,default=180)
    parser.add_argument("--d", type=int, help="Distance apart",required=False,default=0.062)
    parser.add_argument("--rx-gain", type=int, help="RX gain",required=False,default=40)
    parser.add_argument("--tx-gain", type=int, help="TX gain",required=False,default=-3)
    parser.add_argument("--grbl-serial", type=str, help="serial file for GRBL",required=True)
    parser.add_argument("--out", type=str, help="output file",required=True)
    parser.add_argument("--record-freq", type=str, help="record freq",required=False,default=1)
    parser.add_argument("--record-n", type=str, help="records",required=False,default=43200)
    args = parser.parse_args()

    #setup output recorder
    record_matrix = np.memmap(args.out, dtype='float32', mode='w+', shape=(args.record_n,4))

    #setup GRBL
    gm=GRBLManager(args.grbl_serial)

    #calibrate the receiver
    sdr_rx=setup_rxtx_and_phase_calibration(args)
    phase_calibration=sdr_rx.phase_calibration
    sdr_rx=None
    sdr_rx,sdr_tx=setup_rx_and_tx(args)

    #apply the previous calibration
    sdr_rx.phase_calibration=phase_calibration

    #start moving GRBL
    gm_thread = threading.Thread(target=bounce_grbl, args=(gm,))
    gm_thread.start()


    for idx in range(args.record_n):
        print(gm.position)
        time.sleep(1)
        print(get_avg_phase(sdr_rx))
    #x.join()



    #plot_recv_signal(sdr_rx)
    

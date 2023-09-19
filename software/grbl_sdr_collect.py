from sdrpluto.gather import * 
from grbl.grbl_interactive import GRBLManager 
import threading
import time
import numpy as np

def bounce_grbl(gm):
    direction=None
    while True:
        print("TRY TO BOUNCE")
        direction=gm.bounce(1,direction=direction)
        print("TRY TO BOUNCE RET")
        time.sleep(15) # cool off the motor

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-ip", type=str, help="receiver Pluto IP address",required=True)
    parser.add_argument("--emitter-ip", type=str, help="emitter Pluto IP address",required=True)
    parser.add_argument("--fi", type=int, help="Intermediate frequency",required=False,default=1e5)
    parser.add_argument("--fc", type=int, help="Carrier frequency",required=False,default=2.5e9)
    parser.add_argument("--fs", type=int, help="Sampling frequency",required=False,default=16e6)
    parser.add_argument("--cal0", type=int, help="Rx0 calibration phase offset in degrees",required=False,default=180)
    #parser.add_argument("--d", type=int, help="Distance apart",required=False,default=0.062)
    parser.add_argument("--rx-gain", type=int, help="RX gain",required=False,default=40)
    parser.add_argument("--tx-gain", type=int, help="TX gain",required=False,default=-3)
    parser.add_argument("--grbl-serial", type=str, help="serial file for GRBL",required=True)
    parser.add_argument("--out", type=str, help="output file",required=True)
    parser.add_argument("--record-freq", type=int, help="record freq",required=False,default=5)
    parser.add_argument("--rx-mode", type=int, help="rx mode",required=False,default="slow_attack")
    parser.add_argument("--record-n", type=int, help="records",required=False,default=43200)
    parser.add_argument("--rx-n", type=int, help="RX buffer size",required=False,default=2**16)
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

    time_offset=time.time()
    for idx in range(args.record_n):
        while not gm.position['is_moving']:
            print("wait for movement")
            time.sleep(1)
        current_time=time.time()-time_offset
        avg_phase_diff=get_avg_phase(sdr_rx)
        xy=gm.position['xy']
        record_matrix[idx]=np.array([current_time,xy[0],xy[1],avg_phase_diff])
        print(record_matrix[idx])
        time.sleep(1.0/args.record_freq)
    #x.join()



    #plot_recv_signal(sdr_rx)
    

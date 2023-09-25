from sdrpluto.gather import * 
from grbl.grbl_interactive import GRBLManager 
from model_training_and_inference.utils.rf import beamformer
import threading
import time
import numpy as np
import sys
import os
import pickle

def bounce_grbl(gm):
    direction=None
    while gm.collect:
        print("TRY TO BOUNCE")
        try:
            direction=gm.bounce(1,direction=direction)
        except Exception as e:
            print(e)
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
    parser.add_argument("--rx-gain", type=int, help="RX gain",required=False,default=-3)
    parser.add_argument("--tx-gain", type=int, help="TX gain",required=False,default=-3)
    parser.add_argument("--grbl-serial", type=str, help="serial file for GRBL",required=True)
    parser.add_argument("--out", type=str, help="output file",required=True)
    parser.add_argument("--record-freq", type=int, help="record freq",required=False,default=5)
    parser.add_argument("--rx-mode", type=str, help="rx mode",required=False,default="fast_attack")
    parser.add_argument("--record-n", type=int, help="records",required=False,default=43200)
    parser.add_argument("--rx-n", type=int, help="RX buffer size",required=False,default=2**12)
    args = parser.parse_args()

    #setup output recorder
    record_matrix = np.memmap(args.out, dtype='float32', mode='w+', shape=(args.record_n,5+65))

    #setup GRBL
    gm=GRBLManager(args.grbl_serial)

    #calibrate the receiver
    sdr_rx=setup_rxtx_and_phase_calibration(args)
    if sdr_rx==None:
        print("Failed phase calibration, exiting")
        sys.exit(1)
    phase_calibration=sdr_rx.phase_calibration
    sdr_rx=None
    sdr_rx,sdr_tx=setup_rx_and_tx(args)
    if sdr_rx==None or sdr_tx==None:
        print("Failed setup, exiting")
        sys.exit(1)

    #apply the previous calibration
    sdr_rx.phase_calibration=phase_calibration

    #start moving GRBL
    gm_thread = threading.Thread(target=bounce_grbl, args=(gm,))
    gm_thread.start()

    pos=np.array([[-0.03,0],[0.03,0]])

    time_offset=time.time()
    for idx in range(args.record_n):
        while not gm.position['is_moving']:
            print("wait for movement")
            time.sleep(1)

        #get some data
        try:
            signal_matrix=sdr_rx.rx()
        except Exception as e:
            print("Failed to receive RX data! removing file",e)
            os.remove(args.out)
            break
        signal_matrix[1]*=np.exp(1j*sdr_rx.phase_calibration)
        current_time=time.time()-time_offset # timestamp

        beam_thetas,beam_sds,beam_steer=beamformer(
          pos,
          signal_matrix,
          args.fc
        )

        avg_phase_diff=get_avg_phase(signal_matrix)
        xy=gm.position['xy']
        record_matrix[idx]=np.hstack(
                [
                    np.array([current_time,xy[0],xy[1],avg_phase_diff[0],avg_phase_diff[1]]),
                    beam_sds])
        #time.sleep(1.0/args.record_freq)

        if idx%50==0:
            elapsed=time.time()-time_offset
            rate=elapsed/(idx+1)
            print(idx,
                    "mean: %0.4f" % avg_phase_diff[0],
                    "_mean: %0.4f" % avg_phase_diff[1],
                    "%0.4f" % (100.0*float(idx)/args.record_n),
                    '%',
                    "elapsed(min): %0.1f" % (elapsed/60),
                    "rate(s/idx): %0.3f" % rate,
                    "remaining(min): %0.3f" % ((args.record_n-idx)*rate/60))
    gm.collect=False
    print("Done collecting!")
    gm_thread.join()
    #plot_recv_signal(sdr_rx)
    

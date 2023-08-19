import argparse
import adi
import numpy as np
from math import gcd
c=3e8

def setup(args):
    fc0 = int(args.fi)
    fs = int(args.fs)    # must be <=30.72 MHz if both channels are enabled
    rx_lo = int(args.fc) #4e9
    tx_lo = rx_lo

    # setup receive
    rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
    rx_gain = 40 
    tx_gain = -30

    rx_n=int(2**10)

    sdr_receiver = adi.ad9361(uri='ip:%s' % args.receiver_ip)

    sdr_receiver.rx_enabled_channels = [0, 1]
    sdr_receiver.sample_rate = fs
    assert(sdr_receiver.sample_rate==fs)
    sdr_receiver.rx_rf_bandwidth = int(fs) #fc0*5) #TODO!
    sdr_receiver.rx_lo = int(rx_lo)
    sdr_receiver.gain_control_mode = rx_mode
    sdr_receiver.rx_hardwaregain_chan0 = int(rx_gain)
    sdr_receiver.rx_hardwaregain_chan1 = int(rx_gain)
    sdr_receiver.rx_buffer_size = int(rx_n)
    sdr_receiver._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto

    #drop the first bunch of frames
    for _ in range(20):
        sdr_receiver.rx()

    freq = np.fft.fftfreq(rx_n,d=1.0/fs)

    emitter_fs=16e6

    retries=0
    while retries<10:
        #try to setup TX and lets see if it works
        sdr_emitter = adi.ad9361(uri='ip:%s' % args.emitter_ip)

        #setup TX
        sdr_emitter.sample_rate=emitter_fs
        assert(sdr_emitter.sample_rate==emitter_fs)
        sdr_emitter.tx_rf_bandwidth = int(emitter_fs)
        assert(sdr_emitter.tx_rf_bandwidth==int(emitter_fs))
        sdr_emitter.tx_lo = int(tx_lo)
        assert(sdr_emitter.tx_lo==tx_lo)
        sdr_emitter.tx_enabled_channels = [0]
        sdr_emitter.tx_hardwaregain_chan0 = int(-5) #tx_gain) #tx_gain)
        assert(sdr_emitter.tx_hardwaregain_chan0==int(-5))
        sdr_emitter.tx_hardwaregain_chan1 = int(-80) # use Tx2 for calibration
        #
        tx_n=int(fs/gcd(fs,fc0))
        while tx_n<1024*16:
            tx_n*=2
        #sdr_emitter.tx_buffer_size = tx_n

        #since its a cyclic buffer its important to end on a full phase
        t = np.arange(0, tx_n)/fs # time at each point assuming we are sending samples at (1/fs)s
        iq0 = np.exp(1j*2*np.pi*fc0*t)*(2**14)
        #try to reset the tx
        #sdr_emitter.tx_destroy_buffer()
        sdr_emitter.tx_cyclic_buffer = True # this keeps repeating!
        assert(sdr_emitter.tx_cyclic_buffer==True)
        sdr_emitter.tx(iq0)  # Send Tx data.
        
        #give RX a chance to calm down
        for _ in range(50):
            sdr_receiver.rx()

        #test to see what frequency we are seeing
        signal_matrix=np.vstack(sdr_receiver.rx())
        sp = np.fft.fft(signal_matrix[0])
        max_freq=freq[np.abs(np.argmax(sp.real))]
        if np.abs(max_freq-args.fi)<(args.fs/rx_n+1):
            print("TX ONLINE!")
            break
        retries+=1
    return sdr_receiver,sdr_emitter



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
    args = parser.parse_args()
    sdr_rx,sdr_tx=setup(args)

import argparse
import adi
import numpy as np
from math import gcd
import matplotlib.pyplot as plt
c=3e8

def setup_rxtx_and_phase_calibration(args):
    fc0 = int(args.fi)
    fs = int(args.fs)    # must be <=30.72 MHz if both channels are enabled
    rx_lo = int(args.fc) #4e9
    tx_lo = rx_lo

    # setup receive
    rx_mode = "slow_attack" #"slow_attack"  # can be "manual" or "slow_attack"
    rx_gain = args.rx_gain 

    rx_n=int(2**8)

    tx_gain_calibration=-50

    retries=0
    while retries<10:
        #try to setup TX and lets see if it works
        #sdr_emitter = adi.ad9361(uri='ip:%s' % args.emitter_ip)
        sdr_rxtx = adi.ad9361(uri='ip:%s' % args.receiver_ip)

        sdr_rxtx.rx_enabled_channels = [0, 1]
        sdr_rxtx.sample_rate = fs
        assert(sdr_rxtx.sample_rate==fs)
        sdr_rxtx.rx_rf_bandwidth = int(fs) #fc0*5) #TODO!
        sdr_rxtx.rx_lo = int(rx_lo)
        sdr_rxtx.gain_control_mode = rx_mode
        sdr_rxtx.rx_hardwaregain_chan0 = int(rx_gain)
        sdr_rxtx.rx_hardwaregain_chan1 = int(rx_gain)
        sdr_rxtx.rx_buffer_size = int(rx_n)
        sdr_rxtx._rxadc.set_kernel_buffers_count(2)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto

        #drop the first bunch of frames
        for _ in range(20):
            sdr_rxtx.rx()

        #setup TX
        sdr_rxtx.tx_rf_bandwidth = int(fs)
        assert(sdr_rxtx.tx_rf_bandwidth==int(fs))
        sdr_rxtx.tx_lo = int(tx_lo)
        assert(sdr_rxtx.tx_lo==tx_lo)
        sdr_rxtx.tx_enabled_channels = [1]
        sdr_rxtx.tx_hardwaregain_chan0 = int(-80) #tx_gain) #tx_gain)
        sdr_rxtx.tx_hardwaregain_chan1 = int(tx_gain_calibration) # use Tx2 for calibration
        assert(sdr_rxtx.tx_hardwaregain_chan1==int(tx_gain_calibration))
        #
        tx_n=int(fs/gcd(fs,fc0))
        while tx_n<1024*16:
            tx_n*=2
        #sdr_rxtx.tx_buffer_size = tx_n

        #since its a cyclic buffer its important to end on a full phase
        t = np.arange(0, tx_n)/fs # time at each point assuming we are sending samples at (1/fs)s
        iq0 = np.exp(1j*2*np.pi*fc0*t)*(2**14)
        #try to reset the tx
        #sdr_rxtx.tx_destroy_buffer()
        sdr_rxtx.tx_cyclic_buffer = True # this keeps repeating!
        assert(sdr_rxtx.tx_cyclic_buffer==True)
        sdr_rxtx.tx(iq0)  # Send Tx data.
        
        #give RX a chance to calm down
        for _ in range(50):
            sdr_rxtx.rx()

        #test to see what frequency we are seeing
        freq = np.fft.fftfreq(rx_n,d=1.0/fs)
        signal_matrix=np.vstack(sdr_rxtx.rx())
        sp = np.fft.fft(signal_matrix[0])
        max_freq=freq[np.abs(np.argmax(sp.real))]
        if np.abs(max_freq-args.fi)<(args.fs/rx_n+1):
            print("TX ONLINE!")
            break
        retries+=1

    #get some new data
    for retry in range(20):
      n_calibration_frames=200
      phase_calibrations=np.zeros(n_calibration_frames)
      for idx in range(n_calibration_frames):
          sdr_rxtx.rx()
          phase_calibrations[idx]=((np.angle(signal_matrix[0])-np.angle(signal_matrix[1]))%(2*np.pi)).mean() # TODO THIS BREAKS if diff is near 2*np.pi...
      if phase_calibrations.std()<1e-5:
        print("FINAL PHASE CALIBRATION",phase_calibrations.mean(),phase_calibrations.mean()/(2*np.pi))
        sdr_rxtx.phase_calibration=phase_calibrations.mean()
        return sdr_rxtx
    return None



def setup_rx_and_tx(args):
    fc0 = int(args.fi)
    fs = int(args.fs)    # must be <=30.72 MHz if both channels are enabled
    rx_lo = int(args.fc) #4e9
    tx_lo = rx_lo

    # setup receive
    rx_mode = "slow_attack" #"slow_attack"  # can be "manual" or "slow_attack"
    rx_gain = args.rx_gain

    rx_n=int(2**8)

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
    sdr_receiver._rxadc.set_kernel_buffers_count(2)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto

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
        sdr_emitter.tx_hardwaregain_chan0 = int(args.tx_gain) #tx_gain) #tx_gain)
        assert(sdr_emitter.tx_hardwaregain_chan0==int(args.tx_gain))
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


def circular_mean(angles):
    return np.arctan2(np.sin(angles).sum(),np.cos(angles).sum())%(2*np.pi)

def get_avg_phase(sdr_rx):
    rx_n=sdr_rx.rx_buffer_size
    t=np.arange(rx_n)
    signal_matrix=np.vstack(sdr_rx.rx())
    signal_matrix[1]*=np.exp(1j*sdr_rx.phase_calibration)

    diffs=(np.angle(signal_matrix[0])-np.angle(signal_matrix[1]))%(2*np.pi)
    return circular_mean(diffs)

def plot_recv_signal(sdr_rx):
    fig,axs=plt.subplots(2,4,figsize=(16,6))

    rx_n=sdr_rx.rx_buffer_size
    t=np.arange(rx_n)
    while True:
      signal_matrix=np.vstack(sdr_rx.rx())
      signal_matrix[1]*=np.exp(1j*sdr_rx.phase_calibration)

      freq = np.fft.fftfreq(t.shape[-1],d=1.0/sdr_rx.sample_rate)
      assert(t.shape[-1]==rx_n)
      for idx in [0,1]:
        axs[idx][0].clear()
        axs[idx][1].clear()
        axs[idx][0].scatter(t,signal_matrix[idx].real,s=1)
        sp = np.fft.fft(signal_matrix[idx])
        axs[idx][1].scatter(freq, sp.real,s=1) #, freq, sp.imag)
        max_freq=freq[np.abs(np.argmax(sp.real))]
        axs[idx][1].axvline(
          x=max_freq,
          label="max %0.2e" % max_freq,
          color='red'
        )
        axs[idx][1].legend()

        axs[idx][2].clear()
        axs[idx][2].scatter(signal_matrix[idx].real,signal_matrix[idx].imag,s=1)

        axs[idx][0].set_title("Real signal recv (%d)" % idx)
        axs[idx][1].set_title("Power recv (%d)" % idx)
        #print("MAXFREQ",freq[np.abs(np.argmax(sp.real))])
      diff=(np.angle(signal_matrix[0])-np.angle(signal_matrix[1]))%(2*np.pi)
      axs[0][3].clear()
      axs[0][3].scatter(t,diff,s=1)
      axs[0][3].axhline(y = circular_mean(diff))
      axs[0][3].set_ylim([0,2*np.pi])
      fig.canvas.draw()
      plt.pause(0.00001)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-ip", type=str, help="receiver Pluto IP address",required=True)
    parser.add_argument("--emitter-ip", type=str, help="emitter Pluto IP address",required=True)
    parser.add_argument("--fi", type=int, help="Intermediate frequency",required=False,default=1e5)
    parser.add_argument("--fc", type=int, help="Carrier frequency",required=False,default=2.5e9)
    parser.add_argument("--fs", type=int, help="Sampling frequency",required=False,default=16e6)
    parser.add_argument("--cal0", type=int, help="Rx0 calibration phase offset in degrees",required=False,default=180)
    parser.add_argument("--d", type=int, help="Distance apart",required=False,default=0.062)
    parser.add_argument("--rx-gain", type=int, help="RX gain",required=False,default=-3)
    parser.add_argument("--tx-gain", type=int, help="TX gain",required=False,default=-8)
    args = parser.parse_args()

    #calibrate the receiver
    sdr_rx=setup_rxtx_and_phase_calibration(args)
    phase_calibration=sdr_rx.phase_calibration
    sdr_rx=None
    sdr_rx,sdr_tx=setup_rx_and_tx(args)
    #apply the previous calibration
    sdr_rx.phase_calibration=phase_calibration


    plot_recv_signal(sdr_rx)

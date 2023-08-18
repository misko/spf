#Starter code from jon Kraft

import adi
import matplotlib.pyplot as plt
import numpy as np
from math import lcm,gcd
import argparse
from utils.rf import ULADetector, beamformer, beamformer_old, dbfs

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, help="target Pluto IP address",required=True)
parser.add_argument("--fi", type=int, help="Intermediate frequency",required=False,default=5e4)
parser.add_argument("--fc", type=int, help="Intermediate frequency",required=False,default=2.5e9)
parser.add_argument("--fs", type=int, help="Intermediate frequency",required=False,default=16e6)
parser.add_argument("--cal0", type=int, help="Rx0 cal in degree",required=False,default=90)
parser.add_argument("--d", type=int, help="Distance apart",required=False,default=0.062)
args = parser.parse_args()



c=3e8
fc0 = int(args.fi)
fs = int(args.fs)    # must be <=30.72 MHz if both channels are enabled
rx_lo = int(args.fc) #4e9
tx_lo = rx_lo

rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = -5
tx_gain = -5

rx_n=int(2**7)

detector=ULADetector(fs,2,args.d)

sdr = adi.ad9361(uri='ip:%s' % args.ip)

#setup RX
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = fs
assert(sdr.sample_rate==fs)
sdr.rx_rf_bandwidth = int(fs) #fc0*5)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain)
sdr.rx_hardwaregain_chan1 = int(rx_gain)
sdr.rx_buffer_size = int(rx_n)
sdr._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto

sdr.sample_rate=fs
sdr.tx_rf_bandwidth = int(fs)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True # this keeps repeating!
sdr.tx_hardwaregain_chan0 = int(-80) #tx_gain) #tx_gain)
sdr.tx_hardwaregain_chan1 = int(-10) # use Tx2 for calibration

tx_n=int(fs/gcd(fs,fc0))
#tx_n=int(lcm(fc0,fs))
while tx_n<1024*16:
	tx_n*=2
sdr.tx_buffer_size = tx_n

#since its a cyclic buffer its important to end on a full phase
t = np.arange(0, tx_n)/fs # time at each point assuming we are sending samples at (1/fs)s
iq0 = np.exp(1j*2*np.pi*fc0*t)*(2**14)
sdr.tx_enabled_channels = [1]
sdr.tx(iq0)  # Send Tx data.

import time
fig,axs=plt.subplots(1,5,figsize=(20,4))

intervals=2*128+1
counts=np.zeros(intervals-1)

freq = np.fft.fftfreq(rx_n,d=1.0/fs)

i=0
while True:
	signal_matrix=np.vstack(sdr.rx())
	signal_matrix[0]*=np.exp(-1j*(2*np.pi/360)*args.cal0)
	thetas,sds,steering=beamformer(
		detector.all_receiver_pos(), 
	    signal_matrix,
		args.fc,
		spacing=intervals)
	if sds.max()>50:
		#print(time.time(),sds.max(),thetas[sds.argmax()])
		counts[sds.argmax()%(intervals-1)]+=1

		axs[1].cla()
		axs[1].set_xlabel("Time")
		axs[1].set_ylabel("Value")
		axs[1].scatter(np.arange(signal_matrix.shape[1]),signal_matrix[0].real,s=2,label="I")
		axs[1].scatter(np.arange(signal_matrix.shape[1]),signal_matrix[0].imag,s=2,label="Q")
		axs[1].set_title("Receiver 1")
		axs[1].legend(loc=3)

		axs[2].cla()
		axs[2].set_xlabel("Time")
		axs[2].set_ylabel("Value")
		#axs[2].scatter(np.arange(signal_matrix.shape[1]),signal_matrix[1].real,s=2,label="I")
		#axs[2].scatter(np.arange(signal_matrix.shape[1]),signal_matrix[1].imag,s=2,label="Q")
		axs[2].scatter(signal_matrix[0].real,signal_matrix[0].imag,s=2,alpha=0.5,label="RX0")
		axs[2].scatter(signal_matrix[1].real,signal_matrix[1].imag,s=2,alpha=0.5,label="RX1")
		axs[2].legend(loc=3)

		axs[0].cla()
		axs[0].stairs(counts, 360*thetas/(2*np.pi))
		axs[0].set_title("Histogram")
		axs[0].set_xlabel("Theta")
		axs[0].set_ylabel("Count")

		axs[3].cla()
		axs[3].scatter(360*thetas/(2*np.pi),sds,s=0.5)
		axs[3].set_title("Beamformer")
		axs[3].set_xlabel("Theta")
		axs[3].set_ylabel("Signal")

		sp = np.fft.fft(signal_matrix[0])
		axs[4].cla()
		axs[4].set_title("FFT")
		axs[4].scatter(freq, sp.real,s=1) #, freq, sp.imag)
		max_freq=freq[np.abs(np.argmax(sp.real))]
		axs[4].axvline(
			x=max_freq,
			label="max %0.2e" % max_freq,
			color='red'
		)
		axs[4].legend(loc=3)
		#print("MAXFREQ",freq[np.abs(np.argmax(sp.real))])
		plt.draw()
		plt.pause(0.01)

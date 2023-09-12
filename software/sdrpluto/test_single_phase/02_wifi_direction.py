#Starter code from jon Kraft

import adi
import matplotlib.pyplot as plt
import numpy as np
from math import lcm,gcd
import argparse
from utils.rf import ULADetector, beamformer, beamformer_old, dbfs

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, help="target Pluto IP address",required=True)
parser.add_argument("--fi", type=int, help="Intermediate frequency",required=False,default=1e5)
parser.add_argument("--fc", type=int, help="Intermediate frequency",required=False,default=2.5e9)
parser.add_argument("--fs", type=int, help="Intermediate frequency",required=False,default=16e6)
parser.add_argument("--cal0", type=int, help="Rx0 calibration phase offset in degrees",required=False,default=180)
parser.add_argument("--d", type=int, help="Distance apart",required=False,default=0.062)
args = parser.parse_args()



c=3e8
fc0 = int(args.fi)
fs = int(args.fs)    # must be <=30.72 MHz if both channels are enabled
rx_lo = int(args.fc) #4e9
tx_lo = rx_lo


rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
rx_gain = 40 
tx_gain = -30

rx_n=int(2**15)

detector=ULADetector(fs,2,args.d)

sdr = adi.ad9361(uri='ip:%s' % args.ip)

#setup RX
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = fs
assert(sdr.sample_rate==fs)
sdr.rx_rf_bandwidth = int(fc0*3)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain)
sdr.rx_hardwaregain_chan1 = int(rx_gain)
sdr.rx_buffer_size = int(rx_n)
sdr._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto

#setup TX
#sdr.tx_enabled_channels = []

import time
fig,axs=plt.subplots(1,4,figsize=(16,4))

intervals=2*64+1
counts=np.zeros(64)

freq = np.fft.fftfreq(rx_n,d=1.0/fs)

while True:
	signal_matrix=np.vstack(sdr.rx())
	signal_matrix[0]*=np.exp(-1j*(2*np.pi/360)*args.cal0)
	thetas,sds,steering=beamformer(
		detector.all_receiver_pos(), 
	    signal_matrix,
		args.fc,
		spacing=intervals)
	thetas=thetas[64:]
	sds=sds[64:]
	steering=steering[64:]
	if sds.max()>50:
		#print(time.time(),sds.max(),thetas[sds.argmax()])
		counts[sds.argmax()%64]+=1

		axs[1].cla()
		axs[1].set_xlabel("Time")
		axs[1].set_ylabel("Value")
		#axs[2].scatter(np.arange(signal_matrix.shape[1]),signal_matrix[1].real,s=2,label="I")
		#axs[2].scatter(np.arange(signal_matrix.shape[1]),signal_matrix[1].imag,s=2,label="Q")
		axs[1].scatter(signal_matrix[0].real,signal_matrix[0].imag,s=2,alpha=0.5,label="RX1")
		axs[1].scatter(signal_matrix[1].real,signal_matrix[1].imag,s=2,alpha=0.5,label="RX2")
		axs[1].set_title("Receiver signals")
		axs[1].legend(loc=3)

		axs[0].cla()
		axs[0].stairs(counts, 360*thetas/(2*np.pi))
		axs[0].set_title("Histogram")
		axs[0].set_xlabel("Theta")
		axs[0].set_ylabel("Count")

		axs[2].cla()
		axs[2].scatter(360*thetas/(2*np.pi),sds,s=0.5)
		axs[2].set_title("Beamformer")
		axs[2].set_xlabel("Theta")
		axs[2].set_ylabel("Signal")

		sp = np.fft.fft(signal_matrix[0])
		axs[3].cla()
		axs[3].set_title("FFT")
		axs[3].scatter(freq, sp.real,s=1) #, freq, sp.imag)
		max_freq=freq[np.abs(np.argmax(sp.real))]
		axs[3].axvline(
			x=max_freq,
			label="max %0.2e" % max_freq,
			color='red'
		)
		axs[3].legend(loc=3)
		#print("MAXFREQ",freq[np.abs(np.argmax(sp.real))])
		plt.draw()
		plt.pause(0.01)

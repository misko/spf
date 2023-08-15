#Starter code from jon Kraft

import adi
import matplotlib.pyplot as plt
import numpy as np
from math import lcm
#from sdr import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, help="target Pluto IP address",required=True)
args = parser.parse_args()

c=3e8
fc0 = int(1e5)
fs = int(4e6)    # must be <=30.72 MHz if both channels are enabled
rx_lo = int(2.5e9) #4e9
tx_lo = rx_lo

rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = -10 
tx_gain = -30

wavelength = c/rx_lo              # wavelength of the RF carrier

rx_n=int(2**14)

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

#setup TX
sdr.tx_enabled_channels = []
#sdr.tx_rf_bandwidth = int(fc0*3)
#sdr.tx_lo = int(tx_lo)
#sdr.tx_cyclic_buffer = True # this keeps repeating!
#sdr.tx_hardwaregain_chan0 = int(-80) #tx_gain) #tx_gain)
#sdr.tx_hardwaregain_chan1 = int(-80) # use Tx2 for calibration

import time
fig,axs=plt.subplots(1,1,figsize=(4,4))




import matplotlib.pyplot as plt

fig,axs=plt.subplots(2,2,figsize=(10,8))

t=np.arange(rx_n)
while True:
	signal_matrix=np.vstack(sdr.rx())

	freq = np.fft.fftfreq(t.shape[-1],d=1.0/fs)
	assert(t.shape[-1]==rx_n)
	for idx in [0,1]:
		axs[idx][0].clear()
		axs[idx][1].clear()
		axs[idx][0].scatter(t,signal_matrix[idx].real,s=1)
		sp = np.fft.fft(signal_matrix[idx])
		#sp[:int(rx_n/8)]=0
		#sp[-int(rx_n/8):]=0
		axs[idx][1].scatter(freq, sp.real,s=1) #, freq, sp.imag)
		print("MAXFREQ",freq[np.abs(np.argmax(sp.real))])
	fig.canvas.draw()
	plt.pause(0.00001)

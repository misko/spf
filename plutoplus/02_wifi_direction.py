#Starter code from jon Kraft

import adi
#;import matplotlib.pyplot as plt
import numpy as np
from math import lcm
from sdr import *


c=3e8
fc0 = int(500e3)
fs = int(4e6)    # must be <=30.72 MHz if both channels are enabled
rx_lo = int(2.45e9) #4e9
tx_lo = rx_lo

rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = -30 
tx_gain = -30

wavelength = c/rx_lo              # wavelength of the RF carrier

rx_n=int(2**10)

sdr = adi.ad9361(uri='ip:192.168.3.1')

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
sdr.tx_enabled_channels = [0,1]
sdr.tx_rf_bandwidth = int(fc0*3)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True # this keeps repeating!
sdr.tx_hardwaregain_chan0 = int(-88) #tx_gain)
sdr.tx_hardwaregain_chan1 = int(tx_gain) # use Tx2 for calibration
tx_n=int(min(lcm(fc0,fs),rx_n*8)) #1024*1024*1024) # tx for longer than rx
sdr.tx_buffer_size = tx_n*2 #tx_n

#since its a cyclic buffer its important to end on a full phase
t = np.arange(0, tx_n)/fs
iq0 = np.exp(1j*2*np.pi*t*fc0)*(2**14)
sdr.tx([iq0,iq0])  # Send Tx data.

detector=ULADetector(fs,2,wavelength/2)

def sample_phase_offset_rx(iterations=64,calibration=1+0j):
	steerings=[]
	for _ in np.arange(iterations):
		signal_matrix=np.vstack(sdr.rx())
		thetas,sds,steering=beamformer(detector,signal_matrix,rx_lo,spacing=2*1024+1,calibration=calibration)
		steerings.append(steering[sds.argmax()])
	return np.array(steerings).mean(axis=0)

calibration=np.conjugate(sample_phase_offset_rx())
print("calibration complete",calibration)
calibration_new=np.conjugate(sample_phase_offset_rx(calibration=calibration))
print("new cal",calibration_new)

sdr.tx_destroy_buffer()
sdr.rx_destroy_buffer()

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
sdr.tx_enabled_channels = []
#sdr.tx_rf_bandwidth = int(fc0*3)
#sdr.tx_lo = int(tx_lo)
#sdr.tx_cyclic_buffer = True # this keeps repeating!
#sdr.tx_hardwaregain_chan0 = int(-80) #tx_gain) #tx_gain)
#sdr.tx_hardwaregain_chan1 = int(-80) # use Tx2 for calibration
#
#tx_n=int(min(lcm(fc0,fs),rx_n*8)) #1024*1024*1024) # tx for longer than rx
#sdr.tx_buffer_size = tx_n

#since its a cyclic buffer its important to end on a full phase
t = np.arange(0, tx_n)/fs
iq0 = np.exp(1j*2*np.pi*t*fc0)*(2**14)
#sdr.tx(iq0)  # Send Tx data.
import time
fig,axs=plt.subplots(1,1,figsize=(4,4))

intervals=2*64+1
counts=np.zeros(intervals-1)
while True:
	signal_matrix=np.vstack(sdr.rx())
	thetas,sds,steering=beamformer(detector,signal_matrix,rx_lo,spacing=intervals,calibration=calibration)
	if sds.max()>1000:
		print(time.time(),sds.max(),thetas[sds.argmax()])
		counts[sds.argmax()%(intervals-1)]+=1
		axs.cla()
		axs.stairs(counts, 360*thetas/(2*np.pi))
		#axs.scatter(360*thetas/(2*np.pi),sds,s=0.5)
		plt.draw()
		plt.pause(0.01)

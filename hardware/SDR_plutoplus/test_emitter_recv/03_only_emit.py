#Starter code from jon Kraft

import adi
import numpy as np
from math import lcm

c=3e8
fc0 = int(4000)
fs = int(4e6)    # must be <=30.72 MHz if both channels are enabled
rx_lo = int(2.45e9) #4e9
tx_lo = rx_lo

rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = -80 
tx_gain = 0

wavelength = c/rx_lo              # wavelength of the RF carrier

rx_n=int(2**10)

sdr = adi.ad9361(uri='ip:192.168.3.2')

#setup TX
sdr.tx_enabled_channels = [0]
sdr.tx_rf_bandwidth = int(fc0)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True # this keeps repeating!
sdr.tx_hardwaregain_chan0 = int(-10) #tx_gain) #tx_gain)
sdr.tx_hardwaregain_chan1 = int(-80) # use Tx2 for calibration
#
tx_n=int(min(lcm(fc0,fs),rx_n*8)) #1024*1024*1024) # tx for longer than rx
sdr.tx_buffer_size = tx_n

#since its a cyclic buffer its important to end on a full phase
t = np.arange(0, tx_n)/fs
iq0 = np.exp(1j*2*np.pi*t*fc0)*(2**14)
sdr.tx(iq0)  # Send Tx data.

import time
time.sleep(30)

#Starter code from jon Kraft

import adi
import numpy as np
from math import lcm,gcd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, help="target Pluto IP address",required=True)
parser.add_argument("--fi", type=int, help="Intermediate frequency",required=False,default=1e5)
parser.add_argument("--fc", type=int, help="Intermediate frequency",required=False,default=2.5e9)
parser.add_argument("--fs", type=int, help="Intermediate frequency",required=False,default=28e6)
args = parser.parse_args()

c=3e8
fc0 = int(args.fi)
fs = int(args.fs)    # must be <=30.72 MHz if both channels are enabled
rx_lo = int(args.fc) #4e9
tx_lo = rx_lo

rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = -80 
tx_gain = -3

wavelength = c/rx_lo              # wavelength of the RF carrier

rx_n=int(2**10)

sdr = adi.ad9361(uri='ip:%s' % args.ip)

#setup TX
sdr.sample_rate=fs
sdr.tx_rf_bandwidth = int(fs)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True # this keeps repeating!
sdr.tx_hardwaregain_chan0 = int(-5) #tx_gain) #tx_gain)
sdr.tx_hardwaregain_chan1 = int(-80) # use Tx2 for calibration
#
#tx_n=int(min(lcm(fc0,fs),rx_n*8)) #1024*1024*1024) # tx for longer than rx
#tx_n=int(lcm(fc0,fs))
#tx_n=fc0 #*8
tx_n=int(fs/gcd(fs,fc0))
#tx_n=int(lcm(fc0,fs))
while tx_n<1024*16:
	tx_n*=2
print("TX_N",tx_n,fs,fc0)
sdr.tx_buffer_size = tx_n
print(sdr.tx_lo,tx_n,lcm(fc0,fs))

#since its a cyclic buffer its important to end on a full phase
t = np.arange(0, tx_n)/fs # time at each point assuming we are sending samples at (1/fs)s
iq0 = np.exp(1j*2*np.pi*fc0*t)*(2**14)
print(iq0)
sdr.tx_enabled_channels = [0]
sdr.tx(iq0)  # Send Tx data.

import time
time.sleep(1)
time.sleep(1)
print("SLEEP")
time.sleep(120)

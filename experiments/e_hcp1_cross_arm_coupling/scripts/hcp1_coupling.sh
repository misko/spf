#!/bin/sh
# E-HCP1 — bound the tee's cross-arm coupling across frequency.
#
# Mechanism under test: a wave reflecting off one RX port returns to the tee junction and
# enters the OTHER port down its own cable. That path is arm-specific and depends on the
# reflecting arm's gain state, so unlike a common-mode junction shift it does NOT cancel
# in angle(RX1)-angle(RX2) -- it can masquerade as the arm-specific residual A.
#
# Observable: hold one arm's gain fixed and sweep the other's. RSSI is input-referred, so
# a fixed arm on an isolated harness reads constant. Any movement bounds the coupling.
# Amplitude and phase move together on a reflection term, so a dB bound is a degrees bound.
#
# No RF change, no new parts. TX2 stays at full scale, which is ~-57 dBm at the ports.
set -u
PHY=ad9361-phy
DDS=/sys/bus/iio/devices/iio:device3
FIXED=41

rssi() { iio_attr -i -c $PHY "$1" rssi 2>/dev/null | tail -1 | sed 's/ dB//'; }
rxg()  { iio_attr -i -c $PHY voltage$1 hardwaregain "$2" >/dev/null 2>&1; }
txg()  { iio_attr -o -c $PHY voltage1 hardwaregain "$1" >/dev/null 2>&1; }
setlo(){ iio_attr -c $PHY altvoltage0 frequency "$1" >/dev/null 2>&1
         iio_attr -c $PHY altvoltage1 frequency "$1" >/dev/null 2>&1; }

LO0=$(iio_attr -c $PHY altvoltage0 frequency 2>/dev/null | tail -1)
TLO0=$(iio_attr -c $PHY altvoltage1 frequency 2>/dev/null | tail -1)
TX0=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null | tail -1 | sed 's/ dB//')
RAW4=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw); RAW6=$(cat $DDS/out_altvoltage6_TX2_Q_F1_raw)
echo "orig_rx_lo=$LO0"
echo "orig_tx_lo=$TLO0"
echo "orig_tx2_gain=$TX0"
echo "fixed_arm_gain_db=$FIXED"
echo "swept_gains_db=20 30 40 50 60 70"

restore() {
  txg -80
  echo "$RAW4" > $DDS/out_altvoltage4_TX2_I_F1_raw 2>/dev/null
  echo "$RAW6" > $DDS/out_altvoltage6_TX2_Q_F1_raw 2>/dev/null
  setlo "$LO0"
  rxg 0 41; rxg 1 41
  echo "restored_rx_lo=$(iio_attr -c $PHY altvoltage0 frequency 2>/dev/null|tail -1)"
  echo "restored_tx_lo=$(iio_attr -c $PHY altvoltage1 frequency 2>/dev/null|tail -1)"
  echo "restored_tx2_gain=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1)"
  echo "restored_dds_raw=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw)"
  echo "restored_rx_gains=$(iio_attr -i -c $PHY voltage0 hardwaregain 2>/dev/null|tail -1|sed 's/ dB//'):$(iio_attr -i -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1|sed 's/ dB//')"
}
trap restore EXIT INT TERM HUP

echo 1 > $DDS/out_altvoltage4_TX2_I_F1_raw
echo 1 > $DDS/out_altvoltage6_TX2_Q_F1_raw
txg 0

for LO in 433000000 700000000 1000000000 1300000000 \
          1500000000 2400000000 3200000000 4000000000 \
          4300000000 5000000000 5500000000 5900000000; do
  setlo $LO
  sleep 1
  GOT=$(iio_attr -c $PHY altvoltage0 frequency 2>/dev/null | tail -1)
  # sweep RX1, watch RX2 (the fixed arm)
  rxg 1 $FIXED
  for g in 20 30 40 50 60 70; do
    rxg 0 $g; sleep 0.3
    echo "trial=cpl:lo=$GOT:swept=rx1:swept_db=$g:fixed=rx2:fixed_db=$FIXED:rssi_swept=$(rssi voltage0):rssi_fixed=$(rssi voltage1)"
  done
  # sweep RX2, watch RX1
  rxg 0 $FIXED
  for g in 20 30 40 50 60 70; do
    rxg 1 $g; sleep 0.3
    echo "trial=cpl:lo=$GOT:swept=rx2:swept_db=$g:fixed=rx1:fixed_db=$FIXED:rssi_swept=$(rssi voltage1):rssi_fixed=$(rssi voltage0)"
  done
done
echo "coupling_sweep_complete=true"
exit 0

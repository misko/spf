#!/bin/sh
# E-AGC1 step 5, phase C — H3 detector map by differential attribution, plus the
# hold band (open item O-2).
#
# Method: TX2 at full scale (which is only ~-57 dBm at the RX ports through the 30 dB
# pad, so electrically far inside the +2.5 dBm limit), then walk the ADC-referred level
# by sweeping ONE arm's RX gain while holding the other fixed. Bits that move belong to
# the swept channel -- an attribution test that does not depend on the harness having
# port-to-port isolation, which the bare tee does not have.
#
# The hold band comes out as a difference of two gain settings on the same arm, so the
# harness insertion loss cancels.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
DDS=/sys/bus/iio/devices/iio:device3
FIXED=41
LO_G=10
HI_G=72

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base"); OUT0=$((BASE+54))

r()    { iio_reg $PHY "$1" 2>/dev/null | tail -1; }
rssi() { iio_attr -i -c $PHY "$1" rssi 2>/dev/null | tail -1 | sed 's/ dB//'; }
rxg()  { iio_attr -i -c $PHY voltage$1 hardwaregain "$2" >/dev/null 2>&1; }
txg()  { iio_attr -o -c $PHY voltage1 hardwaregain "$1" >/dev/null 2>&1; }
bits() { s=""; i=7; while [ $i -ge 0 ]; do read v < $G/gpio$((OUT0+i))/value; s="$s$v"; i=$((i-1)); done; echo "$s"; }

R35=$(r 0x035); RAW4=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw); RAW6=$(cat $DDS/out_altvoltage6_TX2_Q_F1_raw)
TX0=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1|sed 's/ dB//')
echo "reg_0x035_original=$R35"
echo "tx2_gain_original=$TX0"
echo "thresholds_0x104=$(r 0x104) 0x105=$(r 0x105) 0x108=$(r 0x108) 0x114=$(r 0x114)"
echo "bit_order=CTRL_OUT7,6,5,4,3,2,1,0"
echo "predicted_map=7:CH1_low_power 6:CH1_large_LMT 5:CH1_large_ADC 4:CH1_small_ADC 3:CH2_low_power 2:CH2_large_LMT 1:CH2_large_ADC 0:CH2_small_ADC"

restore() {
  txg -80
  echo "$RAW4" > $DDS/out_altvoltage4_TX2_I_F1_raw 2>/dev/null
  echo "$RAW6" > $DDS/out_altvoltage6_TX2_Q_F1_raw 2>/dev/null
  iio_reg $PHY 0x035 $R35 >/dev/null 2>&1
  rxg 0 41; rxg 1 41
  i=0; while [ $i -le 7 ]; do
    [ -d $G/gpio$((OUT0+i)) ] && echo $((OUT0+i)) > $G/unexport 2>/dev/null
    i=$((i+1)); done
  STILL=0; i=0; while [ $i -le 7 ]; do [ -d $G/gpio$((OUT0+i)) ] && STILL=1; i=$((i+1)); done
  echo "restored_tx2_gain=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1)"
  echo "restored_reg_0x035=$(r 0x035)"
  echo "restored_dds_raw=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw)"
  echo "restored_rx_gains=$(iio_attr -i -c $PHY voltage0 hardwaregain 2>/dev/null|tail -1|sed 's/ dB//'):$(iio_attr -i -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1|sed 's/ dB//')"
  echo "restored_idx=$(( $(r 0x2B0) & 0x7F )):$(( $(r 0x2B5) & 0x7F ))"
  echo "ctrl_out_pins_released=$( [ $STILL -eq 0 ] && echo true || echo false )"
}
trap restore EXIT INT TERM HUP

i=0
while [ $i -le 7 ]; do
  N=$((OUT0+i)); [ -d $G/gpio$N ] || echo $N > $G/export 2>/dev/null
  echo in > $G/gpio$N/direction 2>/dev/null; i=$((i+1))
done
iio_reg $PHY 0x035 0x03 >/dev/null 2>&1
echo "reg_0x035_set=$(r 0x035)"

# tone on, full scale
echo 1 > $DDS/out_altvoltage4_TX2_I_F1_raw
echo 1 > $DDS/out_altvoltage6_TX2_Q_F1_raw
txg 0
sleep 2
echo "tone_on_tx2_gain=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1|sed 's/ dB//')"

# ---- sweep A: RX1 varies, RX2 fixed ----
rxg 1 $FIXED
g=$LO_G
while [ $g -le $HI_G ]; do
  rxg 0 $g; sleep 0.2
  echo "trial=sweepA:swept=rx1:rx1_db=$g:rx2_db=$FIXED:bits=$(bits):rssi1=$(rssi voltage0):rssi2=$(rssi voltage1)"
  g=$((g+1))
done

# ---- sweep B: RX2 varies, RX1 fixed ----
rxg 0 $FIXED
g=$LO_G
while [ $g -le $HI_G ]; do
  rxg 1 $g; sleep 0.2
  echo "trial=sweepB:swept=rx2:rx1_db=$FIXED:rx2_db=$g:bits=$(bits):rssi1=$(rssi voltage0):rssi2=$(rssi voltage1)"
  g=$((g+1))
done
echo "sweeps_complete=true"
exit 0

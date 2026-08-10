#!/bin/sh
# E-AGC1 step 5, phase B — level ramp with CTRL_OUT recorded at every point.
#
# Ramp control is RSSI, not the detector bits: the two low-power bits are asserted at
# rest (no signal), so "any bit high" is not a usable limit. Stop when RSSI comes within
# RSSI_FLOOR dB of full scale. At RX 41 dB the ADC saturates far below the AD9361's
# +2.5 dBm RF pin limit, so this is well inside the damage margin either way.
#
# H3 is read differentially -- which bits CHANGE with level -- so the resting pattern
# does not matter.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
DDS=/sys/bus/iio/devices/iio:device3
RSSI_FLOOR=4
RX_GAIN=${RX_GAIN:-41}

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base"); OUT0=$((BASE+54))

r()    { iio_reg $PHY "$1" 2>/dev/null | tail -1; }
rssi() { iio_attr -i -c $PHY "$1" rssi 2>/dev/null | tail -1 | sed 's/ dB//'; }
idx1() { echo $(( $(r 0x2B0) & 0x7F )); }
idx2() { echo $(( $(r 0x2B5) & 0x7F )); }
txg()  { iio_attr -o -c $PHY voltage1 hardwaregain "$1" >/dev/null 2>&1; }
rxg()  { iio_attr -i -c $PHY voltage$1 hardwaregain "$2" >/dev/null 2>&1; }

R35_ORIG=$(r 0x035)
RAW4=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw)
RAW6=$(cat $DDS/out_altvoltage6_TX2_Q_F1_raw)
echo "reg_0x035_original=$R35_ORIG"
echo "rx_gain_used=$RX_GAIN"

restore() {
  txg -80
  echo "$RAW4" > $DDS/out_altvoltage4_TX2_I_F1_raw 2>/dev/null
  echo "$RAW6" > $DDS/out_altvoltage6_TX2_Q_F1_raw 2>/dev/null
  iio_reg $PHY 0x035 $R35_ORIG >/dev/null 2>&1
  i=0; while [ $i -le 7 ]; do
    [ -d $G/gpio$((OUT0+i)) ] && echo $((OUT0+i)) > $G/unexport 2>/dev/null
    i=$((i+1)); done
  rxg 0 41; rxg 1 41
  echo "restored_tx2_gain=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1)"
  echo "restored_reg_0x035=$(r 0x035)"
  echo "restored_dds_raw=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw)"
  echo "restored_idx=$(idx1):$(idx2)"
  STILL=0; i=0; while [ $i -le 7 ]; do
    [ -d $G/gpio$((OUT0+i)) ] && STILL=1; i=$((i+1)); done
  echo "ctrl_out_pins_released=$( [ $STILL -eq 0 ] && echo true || echo false )"
}
trap restore EXIT INT TERM

i=0
while [ $i -le 7 ]; do
  N=$((OUT0+i))
  [ -d $G/gpio$N ] || echo $N > $G/export 2>/dev/null
  echo in > $G/gpio$N/direction 2>/dev/null
  i=$((i+1))
done
iio_reg $PHY 0x035 0x03 >/dev/null 2>&1
echo "reg_0x035_set=$(r 0x035)"
echo "reg_0x036=$(r 0x036)"
echo "bit_order=CTRL_OUT7,6,5,4,3,2,1,0"

bits() { s=""; i=7; while [ $i -ge 0 ]; do read v < $G/gpio$((OUT0+i))/value; s="$s$v"; i=$((i-1)); done; echo "$s"; }

rxg 0 $RX_GAIN; rxg 1 $RX_GAIN
txg -80; sleep 1
echo "quiescent_bits=$(bits)"
echo "quiescent_rssi=$(rssi voltage0):$(rssi voltage1)"

for TXG in -80 -75 -70 -65 -60 -55 -50 -45 -40 -38 -36 -34 -32 -30 -28 -26 -24 -22 -20 -18 -16 -14 -12 -10; do
  txg $TXG; sleep 1
  B=$(bits); R1=$(rssi voltage0); R2=$(rssi voltage1)
  echo "trial=ramp:tx=$TXG:bits=$B:rssi1=$R1:rssi2=$R2:idx1=$(idx1):idx2=$(idx2)"
  RI=$(echo "$R1" | cut -d. -f1)
  case "$RI" in ''|*[!0-9-]*) RI=99;; esac
  if [ "$RI" -le "$RSSI_FLOOR" ]; then
    echo "ramp_stop_reason=rssi_within_${RSSI_FLOOR}dB_of_full_scale_at_tx_$TXG"; break
  fi
done
echo "ramp_complete=true"
exit 0

#!/bin/sh
# E-AGC1 step 5, phase D — threshold sweep.
#
# For each threshold register value, walk RX1 gain and record the gain at which the
# relevant CTRL_OUT bit changes state. Register semantics are not assumed: if moving a
# register moves its predicted bit's transition and nothing else, that IS the
# identification.
#
# All threshold writes are read-modify-write, preserving bit 7.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
DDS=/sys/bus/iio/devices/iio:device3
LO_G=8
HI_G=72
STEP=2

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base"); OUT0=$((BASE+54))

r()   { iio_reg $PHY "$1" 2>/dev/null | tail -1; }
rxg() { iio_attr -i -c $PHY voltage$1 hardwaregain "$2" >/dev/null 2>&1; }
txg() { iio_attr -o -c $PHY voltage1 hardwaregain "$1" >/dev/null 2>&1; }
bits(){ s=""; i=7; while [ $i -ge 0 ]; do read v < $G/gpio$((OUT0+i))/value; s="$s$v"; i=$((i-1)); done; echo "$s"; }

R35=$(r 0x035)
T104=$(r 0x104); T105=$(r 0x105); T107=$(r 0x107); T108=$(r 0x108); T114=$(r 0x114)
RAW4=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw); RAW6=$(cat $DDS/out_altvoltage6_TX2_Q_F1_raw)
echo "orig_0x104=$T104 orig_0x105=$T105 orig_0x107=$T107 orig_0x108=$T108 orig_0x114=$T114"
echo "orig_0x035=$R35"
echo "bit_order=CTRL_OUT7,6,5,4,3,2,1,0"

restore() {
  txg -80
  echo "$RAW4" > $DDS/out_altvoltage4_TX2_I_F1_raw 2>/dev/null
  echo "$RAW6" > $DDS/out_altvoltage6_TX2_Q_F1_raw 2>/dev/null
  iio_reg $PHY 0x104 $T104 >/dev/null 2>&1
  iio_reg $PHY 0x105 $T105 >/dev/null 2>&1
  iio_reg $PHY 0x107 $T107 >/dev/null 2>&1
  iio_reg $PHY 0x108 $T108 >/dev/null 2>&1
  iio_reg $PHY 0x114 $T114 >/dev/null 2>&1
  iio_reg $PHY 0x035 $R35 >/dev/null 2>&1
  rxg 0 41; rxg 1 41
  i=0; while [ $i -le 7 ]; do
    [ -d $G/gpio$((OUT0+i)) ] && echo $((OUT0+i)) > $G/unexport 2>/dev/null; i=$((i+1)); done
  echo "restored_0x104=$(r 0x104) restored_0x105=$(r 0x105) restored_0x107=$(r 0x107) restored_0x108=$(r 0x108) restored_0x114=$(r 0x114)"
  echo "restored_0x035=$(r 0x035)"
  echo "restored_tx2=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1)"
  echo "restored_dds_raw=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw)"
  STILL=0; i=0; while [ $i -le 7 ]; do [ -d $G/gpio$((OUT0+i)) ] && STILL=1; i=$((i+1)); done
  echo "ctrl_out_pins_released=$( [ $STILL -eq 0 ] && echo true || echo false )"
}
trap restore EXIT INT TERM HUP

i=0
while [ $i -le 7 ]; do
  N=$((OUT0+i)); [ -d $G/gpio$N ] || echo $N > $G/export 2>/dev/null
  echo in > $G/gpio$N/direction 2>/dev/null; i=$((i+1))
done
iio_reg $PHY 0x035 0x03 >/dev/null 2>&1
echo 1 > $DDS/out_altvoltage4_TX2_I_F1_raw
echo 1 > $DDS/out_altvoltage6_TX2_Q_F1_raw
txg 0
rxg 1 41
sleep 2

# walk RX1 and report the first gain at which bit position $1 (0..7 from the right of the
# printed string, i.e. CTRL_OUT index) reaches state $2
find_edge() {   # $1=ctrl_out index, $2=target state
  idx=$1; want=$2; pos=$(( 7 - idx ))
  g=$LO_G
  while [ $g -le $HI_G ]; do
    rxg 0 $g; sleep 0.2
    B=$(bits)
    c=$(echo "$B" | cut -c$((pos+1)))
    if [ "$c" = "$want" ]; then echo "$g"; return 0; fi
    g=$((g+STEP))
  done
  echo "none"
}

# 0x114 -> predicted CH1 low power (CTRL_OUT7): find gain where it goes 0
for v in 0x10 0x20 0x30 0x40 0x50 0x60; do
  N=$(( ($T114 & 0x80) | ($v & 0x7F) ))
  iio_reg $PHY 0x114 $N >/dev/null 2>&1
  echo "trial=thr:reg=0x114:wrote=$v:readback=$(r 0x114):bit=CTRL_OUT7_CH1_low_power:target=0:edge_gain_db=$(find_edge 7 0)"
done
iio_reg $PHY 0x114 $T114 >/dev/null 2>&1

# 0x107 -> predicted CH1 small ADC overload (CTRL_OUT4): find gain where it goes 1
for v in 0x1b 0x23 0x2b 0x33 0x3b 0x43; do
  N=$(( ($T107 & 0x80) | ($v & 0x7F) ))
  iio_reg $PHY 0x107 $N >/dev/null 2>&1
  echo "trial=thr:reg=0x107:wrote=$v:readback=$(r 0x107):bit=CTRL_OUT4_CH1_small_ADC:target=1:edge_gain_db=$(find_edge 4 1)"
done
iio_reg $PHY 0x107 $T107 >/dev/null 2>&1

# 0x108 -> predicted CH1 large ADC overload (CTRL_OUT5): find gain where it goes 1
for v in 0x21 0x29 0x31 0x39 0x41 0x49; do
  N=$(( ($T108 & 0x80) | ($v & 0x7F) ))
  iio_reg $PHY 0x108 $N >/dev/null 2>&1
  echo "trial=thr:reg=0x108:wrote=$v:readback=$(r 0x108):bit=CTRL_OUT5_CH1_large_ADC:target=1:edge_gain_db=$(find_edge 5 1)"
done
iio_reg $PHY 0x108 $T108 >/dev/null 2>&1

echo "threshold_sweep_complete=true"
exit 0

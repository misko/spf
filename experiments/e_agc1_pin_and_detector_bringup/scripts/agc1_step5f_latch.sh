#!/bin/sh
# E-AGC1 step 5, phase F — is the large-overload bit a LATCH or a LEVEL?
#
# This is H4's first half, and it needs no fast sampling: "stays high until the gain
# changes" is a question about state persistence, answerable with slow reads. Only H4's
# second half -- the sub-millisecond post-change blank -- needs microsecond timing, and
# that is what remains out of reach. Separating them was the mistake in phase E.
#
# Prompted by R18's 0x108 sweep reading asserted at the BOTTOM of the gain range for all
# six register values, where R17 tracked cleanly at 46 dB.
#
# Test: drive into overload, then drop the gain far below the trip point and read. A level
# clears; a latch does not. Then re-drive and re-drop to show it is repeatable, and finally
# check whether anything other than a gain change clears it.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
DDS=/sys/bus/iio/devices/iio:device3
LOWG=20
HIGHG=60

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base"); OUT0=$((BASE+54))

r()   { iio_reg $PHY "$1" 2>/dev/null | tail -1; }
rxg() { iio_attr -i -c $PHY voltage$1 hardwaregain "$2" >/dev/null 2>&1; }
txg() { iio_attr -o -c $PHY voltage1 hardwaregain "$1" >/dev/null 2>&1; }
bits(){ s=""; i=7; while [ $i -ge 0 ]; do read v < $G/gpio$((OUT0+i))/value; s="$s$v"; i=$((i-1)); done; echo "$s"; }
b5()  { read v < $G/gpio$((OUT0+5))/value; echo "$v"; }   # CH1 large ADC overload
b4()  { read v < $G/gpio$((OUT0+4))/value; echo "$v"; }   # CH1 small ADC overload

R35=$(r 0x035); RAW4=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw); RAW6=$(cat $DDS/out_altvoltage6_TX2_Q_F1_raw)
echo "orig_0x035=$R35"
echo "pwot_0x0FE_low5=$(( $(r 0x0FE) & 0x1F ))"

restore() {
  txg -80
  echo "$RAW4" > $DDS/out_altvoltage4_TX2_I_F1_raw 2>/dev/null
  echo "$RAW6" > $DDS/out_altvoltage6_TX2_Q_F1_raw 2>/dev/null
  iio_reg $PHY 0x035 $R35 >/dev/null 2>&1
  rxg 0 41; rxg 1 41
  i=0; while [ $i -le 7 ]; do
    [ -d $G/gpio$((OUT0+i)) ] && echo $((OUT0+i)) > $G/unexport 2>/dev/null; i=$((i+1)); done
  S=0; i=0; while [ $i -le 7 ]; do [ -d $G/gpio$((OUT0+i)) ] && S=1; i=$((i+1)); done
  echo "restored_0x035=$(r 0x035)"
  echo "restored_tx2=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1)"
  echo "restored_dds_raw=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw)"
  echo "pins_released=$( [ $S -eq 0 ] && echo true || echo false )"
}
trap restore EXIT INT TERM HUP

i=0; while [ $i -le 7 ]; do
  N=$((OUT0+i)); [ -d $G/gpio$N ] || echo $N > $G/export 2>/dev/null
  echo in > $G/gpio$N/direction 2>/dev/null; i=$((i+1)); done
iio_reg $PHY 0x035 0x03 >/dev/null 2>&1
echo 1 > $DDS/out_altvoltage4_TX2_I_F1_raw
echo 1 > $DDS/out_altvoltage6_TX2_Q_F1_raw
txg 0; rxg 1 41; sleep 2

# A. establish the clean low state from a cold start
rxg 0 $LOWG; sleep 1
echo "trial=latch:phase=A_cold_low:gain=$LOWG:large=$(b5):small=$(b4):bits=$(bits)"

# B. drive into overload
rxg 0 $HIGHG; sleep 1
echo "trial=latch:phase=B_overload:gain=$HIGHG:large=$(b5):small=$(b4):bits=$(bits)"

# C. drop far below the trip point -- a LEVEL clears here, a LATCH does not
rxg 0 $LOWG; sleep 1
echo "trial=latch:phase=C_dropped:gain=$LOWG:large=$(b5):small=$(b4):bits=$(bits)"

# D. wait longer, in case it is slow rather than sticky
sleep 3
echo "trial=latch:phase=D_after_3s:gain=$LOWG:large=$(b5):small=$(b4):bits=$(bits)"

# E. repeat the cycle to show whatever happens is repeatable
k=1
while [ $k -le 3 ]; do
  rxg 0 $HIGHG; sleep 0.6; HI=$(b5)
  rxg 0 $LOWG;  sleep 0.6; LO=$(b5)
  echo "trial=latch:phase=E_cycle:k=$k:large_at_high=$HI:large_at_low=$LO"
  k=$((k+1))
done

# F. does removing the SIGNAL clear it, as opposed to changing the gain?
rxg 0 $HIGHG; sleep 0.6
echo "trial=latch:phase=F_overload_again:gain=$HIGHG:large=$(b5)"
txg -80; sleep 1
echo "trial=latch:phase=F_tone_muted:gain=$HIGHG:large=$(b5):small=$(b4)"
txg 0; sleep 1
echo "trial=latch:phase=F_tone_back:gain=$HIGHG:large=$(b5)"

# G. and does a small gain step clear it, or only a large one?
rxg 0 $HIGHG; sleep 0.6
echo "trial=latch:phase=G_base:gain=$HIGHG:large=$(b5)"
rxg 0 $((HIGHG-1)); sleep 0.6
echo "trial=latch:phase=G_minus1:gain=$((HIGHG-1)):large=$(b5)"
rxg 0 $((HIGHG-5)); sleep 0.6
echo "trial=latch:phase=G_minus5:gain=$((HIGHG-5)):large=$(b5)"
echo "latch_test_complete=true"
exit 0

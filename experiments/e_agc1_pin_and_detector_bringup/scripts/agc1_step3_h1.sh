#!/bin/sh
# E-AGC1 step 3 — H1 pin mapping, and step 4 part A (step size from the H1 traces).
#
# Arms 0x0FB[1:0] by read-modify-write, pulses each CTRL_IN pin 5x recording BOTH
# gain indices every time, then always disarms via an EXIT trap.
#
# Preconditions (checked): pins already exported, direction out, value 0.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
REPEATS=5
HOME_IDX_DB=41          # index 44 at 868 MHz -- mid table, headroom both ways

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base")

idx1() { echo $(( $(iio_reg $PHY 0x2B0 | tail -1) & 0x7F )); }
idx2() { echo $(( $(iio_reg $PHY 0x2B5 | tail -1) & 0x7F )); }

FB_ORIG=$(iio_reg $PHY 0x0FB | tail -1)
echo "reg_0x0FB_original=$FB_ORIG"

disarm() {
  V=$(iio_reg $PHY 0x0FB | tail -1)
  NEW=$(( $V & ~0x3 & 0xFF ))
  iio_reg $PHY 0x0FB $NEW >/dev/null 2>&1
  AFTER=$(iio_reg $PHY 0x0FB | tail -1)
  echo "disarm_wrote=0x$(printf '%02x' $NEW)"
  echo "reg_0x0FB_after_disarm=$AFTER"
  echo "disarm_ok=$( [ $(( $AFTER & 0x3 )) -eq 0 ] && echo true || echo false )"
  echo "reg_0x0FB_restored_exactly=$( [ "$AFTER" = "$FB_ORIG" ] && echo true || echo false )"
}
trap disarm EXIT INT TERM

# --- verify preconditions: all four pins ours, out, low ---
PRE=0
for off in 62 63 64 65; do
  D=$G/gpio$((BASE+off))
  [ -d "$D" ] || PRE=1
  [ "$(cat $D/direction 2>/dev/null)" = "out" ] || PRE=1
  [ "$(cat $D/value 2>/dev/null)" = "0" ] || PRE=1
done
echo "precondition_pins_low=$( [ $PRE -eq 0 ] && echo true || echo false )"
if [ $PRE -ne 0 ]; then echo "abort=preconditions_not_met"; exit 3; fi

# --- arm, read-modify-write, never a bare 0x03 ---
ARM=$(( $FB_ORIG | 0x3 ))
iio_reg $PHY 0x0FB $ARM >/dev/null 2>&1
FB_ARMED=$(iio_reg $PHY 0x0FB | tail -1)
echo "arm_wrote=0x$(printf '%02x' $ARM)"
echo "reg_0x0FB_armed=$FB_ARMED"
echo "armed_ok=$( [ $(( $FB_ARMED & 0x3 )) -eq 3 ] && echo true || echo false )"
echo "rmw_preserved_other_bits=$( [ $(( $FB_ARMED & ~0x3 & 0xFF )) -eq $(( $FB_ORIG & ~0x3 & 0xFF )) ] && echo true || echo false )"
if [ $(( $FB_ARMED & 0x3 )) -ne 3 ]; then echo "abort=arm_failed"; exit 4; fi

# --- H1: one pin at a time, 5 pulses, both indices every time ---
for off in 62 63 64 65; do
  PIN=$((off-62))
  D=$G/gpio$((BASE+off))

  # return both arms to a known mid-table point before each pin
  iio_attr -i -c $PHY voltage0 hardwaregain $HOME_IDX_DB >/dev/null 2>&1
  iio_attr -i -c $PHY voltage1 hardwaregain $HOME_IDX_DB >/dev/null 2>&1
  echo "home_ctrl_in$PIN=$(idx1):$(idx2)"

  n=1
  while [ $n -le $REPEATS ]; do
    B1=$(idx1); B2=$(idx2)
    echo 1 > $D/value
    echo 0 > $D/value
    A1=$(idx1); A2=$(idx2)
    echo "trial=ctrl_in$PIN:gpio$((BASE+off)):n=$n:rx1_before=$B1:rx1_after=$A1:d1=$((A1-B1)):rx2_before=$B2:rx2_after=$A2:d2=$((A2-B2))"
    n=$((n+1))
  done
done

# restore gains to the baseline point before disarming
iio_attr -i -c $PHY voltage0 hardwaregain $HOME_IDX_DB >/dev/null 2>&1
iio_attr -i -c $PHY voltage1 hardwaregain $HOME_IDX_DB >/dev/null 2>&1
echo "final_idx=$(idx1):$(idx2)"
exit 0

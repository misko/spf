#!/bin/sh
# E-AGC1 step 4 — step size, plus an unplanned A/B found during H1:
# does a software `hardwaregain` write still move the index while pin control is armed?
#
# Restores 0x0FC, 0x0FE and 0x0FB via an EXIT trap.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
HOME_DB=41      # index 44 at 868 MHz
PROBE_DB=35     # index 38 -- a 6 dB / 6 index move, unambiguous

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base")
IN0=$G/gpio$((BASE+62)); IN1=$G/gpio$((BASE+63))

idx1() { echo $(( $(iio_reg $PHY 0x2B0 | tail -1) & 0x7F )); }
idx2() { echo $(( $(iio_reg $PHY 0x2B5 | tail -1) & 0x7F )); }
setg() { iio_attr -i -c $PHY voltage0 hardwaregain "$1" >/dev/null 2>&1; echo "setg_rc=$?"; }

FB0=$(iio_reg $PHY 0x0FB | tail -1)
FC0=$(iio_reg $PHY 0x0FC | tail -1)
FE0=$(iio_reg $PHY 0x0FE | tail -1)
echo "reg_0x0FB_original=$FB0"
echo "reg_0x0FC_original=$FC0"
echo "reg_0x0FE_original=$FE0"

restore() {
  V=$(iio_reg $PHY 0x0FB | tail -1); iio_reg $PHY 0x0FB $(( $V & ~0x3 & 0xFF )) >/dev/null 2>&1
  iio_reg $PHY 0x0FC $FC0 >/dev/null 2>&1
  iio_reg $PHY 0x0FE $FE0 >/dev/null 2>&1
  iio_attr -i -c $PHY voltage0 hardwaregain $HOME_DB >/dev/null 2>&1
  iio_attr -i -c $PHY voltage1 hardwaregain $HOME_DB >/dev/null 2>&1
  echo "restored_0x0FB=$(iio_reg $PHY 0x0FB | tail -1)"
  echo "restored_0x0FC=$(iio_reg $PHY 0x0FC | tail -1)"
  echo "restored_0x0FE=$(iio_reg $PHY 0x0FE | tail -1)"
  echo "restored_idx=$(idx1):$(idx2)"
}
trap restore EXIT INT TERM

# ---------- A/B: hardwaregain write, DISARMED ----------
echo "ab_phase=disarmed"
echo "ab_disarmed_armbits=$(( $(iio_reg $PHY 0x0FB | tail -1) & 0x3 ))"
setg $HOME_DB
echo "ab_disarmed_idx_at_home=$(idx1)"
setg $PROBE_DB
echo "ab_disarmed_idx_after_probe_write=$(idx1)"
setg $HOME_DB
echo "ab_disarmed_idx_back_home=$(idx1)"

# ---------- A/B: hardwaregain write, ARMED ----------
ARM=$(( $FB0 | 0x3 ))
iio_reg $PHY 0x0FB $ARM >/dev/null 2>&1
echo "ab_phase=armed"
echo "ab_armed_armbits=$(( $(iio_reg $PHY 0x0FB | tail -1) & 0x3 ))"
echo "ab_armed_idx_before=$(idx1)"
setg $PROBE_DB
echo "ab_armed_idx_after_probe_write=$(idx1)"
echo "ab_armed_hardwaregain_readback=$(iio_attr -i -c $PHY voltage0 hardwaregain 2>/dev/null | tail -1)"

# does a pin edge still work in this state?
echo 1 > $IN0/value; echo 0 > $IN0/value
echo "ab_armed_idx_after_one_up_pulse=$(idx1)"

# ---------- step 4: program step 1 in both directions ----------
FC1=$(( $FC0 & 0x1F ))          # bits[7:5] = 0 -> increment step 1
FE1=$(( $FE0 & 0x1F ))          # bits[7:5] = 0 -> decrement step 1, PWOT preserved
iio_reg $PHY 0x0FC $FC1 >/dev/null 2>&1
iio_reg $PHY 0x0FE $FE1 >/dev/null 2>&1
echo "step1_wrote_0x0FC=0x$(printf '%02x' $FC1)"
echo "step1_wrote_0x0FE=0x$(printf '%02x' $FE1)"
echo "step1_readback_0x0FC=$(iio_reg $PHY 0x0FC | tail -1)"
echo "step1_readback_0x0FE=$(iio_reg $PHY 0x0FE | tail -1)"
echo "step1_pwot_preserved=$( [ $(( $(iio_reg $PHY 0x0FE | tail -1) & 0x1F )) -eq $(( $FE0 & 0x1F )) ] && echo true || echo false )"

n=1
while [ $n -le 5 ]; do
  B=$(idx1); echo 1 > $IN0/value; echo 0 > $IN0/value; A=$(idx1)
  echo "trial=step1_up:n=$n:before=$B:after=$A:delta=$((A-B))"
  n=$((n+1))
done
n=1
while [ $n -le 5 ]; do
  B=$(idx1); echo 1 > $IN1/value; echo 0 > $IN1/value; A=$(idx1)
  echo "trial=step1_down:n=$n:before=$B:after=$A:delta=$((A-B))"
  n=$((n+1))
done
exit 0

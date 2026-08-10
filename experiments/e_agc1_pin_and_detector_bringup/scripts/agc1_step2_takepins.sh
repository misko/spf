#!/bin/sh
# E-AGC1 step 2 (part A) — take the CTRL_IN pins and verify they read low.
# Deliberately does NOT arm 0x0FB. Arming happens only after this verifies.
# Safe while unarmed: with 0x0FB[1:0]==0 the AD9361 ignores CTRL_IN entirely.
set -u
PHY=ad9361-phy
G=/sys/class/gpio

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base")
echo "gpio_base=$BASE"

# refuse to proceed if pin control is somehow already armed
FB=$(iio_reg $PHY 0x0FB 2>/dev/null | tail -1)
ARMED=$(( $FB & 0x3 ))
echo "reg_0x0FB_before=$FB"
echo "armed_before=$ARMED"
if [ "$ARMED" -ne 0 ]; then
  echo "abort=pin_control_already_armed"
  exit 3
fi

# gain indices before touching anything
echo "idx_rx1_before=$(( $(iio_reg $PHY 0x2B0 | tail -1) & 0x7F ))"
echo "idx_rx2_before=$(( $(iio_reg $PHY 0x2B5 | tail -1) & 0x7F ))"

FAIL=0
for off in 62 63 64 65; do
  N=$((BASE+off))
  D=$G/gpio$N
  if [ ! -d "$D" ]; then
    echo "$N" > $G/export 2>/dev/null || { echo "export_fail=$N"; FAIL=1; continue; }
  else
    echo "already_exported=$N"
  fi
  echo out > $D/direction 2>/dev/null || { echo "direction_fail=$N"; FAIL=1; continue; }
  echo 0   > $D/value     2>/dev/null || { echo "value_fail=$N"; FAIL=1; continue; }
done

# read back and verify -- this is the gate
sleep 1
for off in 62 63 64 65; do
  N=$((BASE+off))
  D=$G/gpio$N
  DIR=$(cat $D/direction 2>/dev/null)
  VAL=$(cat $D/value 2>/dev/null)
  echo "pin_ctrl_in$((off-62))=gpio$N:dir=$DIR:val=$VAL"
  [ "$DIR" = "out" ] || FAIL=1
  [ "$VAL" = "0" ]   || FAIL=1
done

# indices must not have moved: unarmed pins cannot change gain
echo "idx_rx1_after=$(( $(iio_reg $PHY 0x2B0 | tail -1) & 0x7F ))"
echo "idx_rx2_after=$(( $(iio_reg $PHY 0x2B5 | tail -1) & 0x7F ))"

echo "all_four_driven_low=$( [ $FAIL -eq 0 ] && echo true || echo false )"
echo "safe_to_arm=$( [ $FAIL -eq 0 ] && echo true || echo false )"
exit $FAIL

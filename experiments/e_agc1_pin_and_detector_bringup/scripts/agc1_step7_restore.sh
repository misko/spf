#!/bin/sh
# E-AGC1 section 5.4 restore — disarm (already), restore gains, release the pins,
# and prove the part is back where step 1 found it.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
HOME_DB=41

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base")

# 1. disarm first -- must happen BEFORE the pins are released
V=$(iio_reg $PHY 0x0FB | tail -1)
iio_reg $PHY 0x0FB $(( $V & ~0x3 & 0xFF )) >/dev/null 2>&1
FB=$(iio_reg $PHY 0x0FB | tail -1)
echo "disarmed_0x0FB=$FB"
echo "disarmed_ok=$( [ $(( $FB & 0x3 )) -eq 0 ] && echo true || echo false )"

# 2. restore gain (works now that pin control is released -- see the step 4 A/B)
iio_attr -i -c $PHY voltage0 hardwaregain $HOME_DB >/dev/null 2>&1
iio_attr -i -c $PHY voltage1 hardwaregain $HOME_DB >/dev/null 2>&1
echo "gain_restored_idx=$(( $(iio_reg $PHY 0x2B0 | tail -1) & 0x7F )):$(( $(iio_reg $PHY 0x2B5 | tail -1) & 0x7F ))"

# 3. release the pins, leaving them high-Z as found
for off in 62 63 64 65; do
  N=$((BASE+off))
  if [ -d $G/gpio$N ]; then
    echo "$N" > $G/unexport 2>/dev/null && echo "unexported=$N" || echo "unexport_fail=$N"
  else
    echo "already_unexported=$N"
  fi
done
STILL=0
for off in 62 63 64 65; do
  [ -d $G/gpio$((BASE+off)) ] && STILL=1
done
echo "all_pins_released=$( [ $STILL -eq 0 ] && echo true || echo false )"

# 4. no CTRL line may appear as claimed again
echo "claimed_lines_after=$(sed -n 's/^ *gpio-\([0-9]*\).*/\1/p' /sys/kernel/debug/gpio | tr '\n' ',')"

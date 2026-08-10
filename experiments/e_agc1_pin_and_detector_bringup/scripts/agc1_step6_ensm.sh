#!/bin/sh
# E-AGC1 step 6 — H6: are CTRL_IN edges honoured while the ENSM is outside RX?
#
# Tests alert, wait, sleep. Deliberately does NOT touch pinctrl or
# pinctrl_fdd_indep: both hand ENSM state to external pins, which would
# confound the ENSM question with a second pin-control surface (see section 8).
#
# Restores ENSM and disarms via an EXIT trap; aborts if an ENSM restore fails.
set -u
PHY=ad9361-phy
G=/sys/class/gpio

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base")
IN0=$G/gpio$((BASE+62))

idx1() { echo $(( $(iio_reg $PHY 0x2B0 | tail -1) & 0x7F )); }
ensm() { iio_attr -d $PHY ensm_mode 2>/dev/null | tail -1; }
setensm() { iio_attr -d $PHY ensm_mode "$1" >/dev/null 2>&1; echo "rc=$?"; }

ENSM0=$(ensm)
FB0=$(iio_reg $PHY 0x0FB | tail -1)
echo "ensm_original=$ENSM0"
echo "reg_0x0FB_original=$FB0"

restore() {
  iio_attr -d $PHY ensm_mode "$ENSM0" >/dev/null 2>&1
  V=$(iio_reg $PHY 0x0FB | tail -1); iio_reg $PHY 0x0FB $(( $V & ~0x3 & 0xFF )) >/dev/null 2>&1
  echo "restored_ensm=$(ensm)"
  echo "restored_0x0FB=$(iio_reg $PHY 0x0FB | tail -1)"
  echo "restore_ok=$( [ "$(ensm)" = "$ENSM0" ] && echo true || echo false )"
}
trap restore EXIT INT TERM

# arm once; ENSM is what varies
ARM=$(( $FB0 | 0x3 ))
iio_reg $PHY 0x0FB $ARM >/dev/null 2>&1
echo "armed_ok=$( [ $(( $(iio_reg $PHY 0x0FB | tail -1) & 0x3 )) -eq 3 ] && echo true || echo false )"

# control: in the baseline ENSM state, an edge must move the index
B=$(idx1); echo 1 > $IN0/value; echo 0 > $IN0/value; A=$(idx1)
echo "trial=ensm_$ENSM0:state=$ENSM0:before=$B:after=$A:delta=$((A-B)):honoured=$( [ $((A-B)) -ne 0 ] && echo true || echo false )"

for st in alert wait sleep; do
  echo "target_state=$st"
  echo "setensm_$st=$(setensm $st)"
  GOT=$(ensm)
  echo "reached_$st=$GOT"
  if [ "$GOT" != "$st" ]; then
    echo "trial=ensm_$st:state=$GOT:skipped=state_not_reached"
  else
    n=1
    while [ $n -le 3 ]; do
      B=$(idx1); echo 1 > $IN0/value; echo 0 > $IN0/value; A=$(idx1)
      echo "trial=ensm_$st:state=$st:n=$n:before=$B:after=$A:delta=$((A-B)):honoured=$( [ $((A-B)) -ne 0 ] && echo true || echo false )"
      n=$((n+1))
    done
  fi
  # back to baseline state and confirm before moving on
  iio_attr -d $PHY ensm_mode "$ENSM0" >/dev/null 2>&1
  BACK=$(ensm)
  echo "returned_from_$st=$BACK"
  if [ "$BACK" != "$ENSM0" ]; then echo "abort=ensm_restore_failed_after_$st"; exit 5; fi
  # and confirm the part still responds in the baseline state
  B=$(idx1); echo 1 > $IN0/value; echo 0 > $IN0/value; A=$(idx1)
  echo "recheck_after_$st=delta=$((A-B))"
done
exit 0

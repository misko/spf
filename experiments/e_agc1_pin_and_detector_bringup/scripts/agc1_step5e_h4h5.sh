#!/bin/sh
# E-AGC1 step 5, phase E — H4 (latch / post-change blank) and H5 (low-power update rate).
#
# Both are pre-declared resolution-limited (plan section 2): the fastest sysfs read here
# is ~134 us via a shell builtin, against a predicted 256-410 us period. Nothing below
# ~500 us is reportable as measured.
#
# H4 needs the gain to change and then CTRL_OUT sampled within microseconds. An
# iio_attr write costs 67 ms, which is far too slow, so the gain is stepped with the
# CTRL_IN pins validated by H1 -- a GPIO write is a shell builtin at ~134 us.
set -u
PHY=ad9361-phy
G=/sys/class/gpio
DDS=/sys/bus/iio/devices/iio:device3
NSAMP=60

CHIP=""
for c in $G/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base"); OUT0=$((BASE+54)); IN0=$((BASE+62))

r()   { iio_reg $PHY "$1" 2>/dev/null | tail -1; }
rxg() { iio_attr -i -c $PHY voltage$1 hardwaregain "$2" >/dev/null 2>&1; }
txg() { iio_attr -o -c $PHY voltage1 hardwaregain "$1" >/dev/null 2>&1; }
bits(){ s=""; i=7; while [ $i -ge 0 ]; do read v < $G/gpio$((OUT0+i))/value; s="$s$v"; i=$((i-1)); done; echo "$s"; }

R35=$(r 0x035); FB=$(r 0x0FB)
RAW4=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw); RAW6=$(cat $DDS/out_altvoltage6_TX2_Q_F1_raw)
echo "orig_0x035=$R35 orig_0x0FB=$FB orig_pwot=$(( $(r 0x0FE) & 0x1F ))"

restore() {
  V=$(r 0x0FB); iio_reg $PHY 0x0FB $(( $V & ~0x3 & 0xFF )) >/dev/null 2>&1
  txg -80
  echo "$RAW4" > $DDS/out_altvoltage4_TX2_I_F1_raw 2>/dev/null
  echo "$RAW6" > $DDS/out_altvoltage6_TX2_Q_F1_raw 2>/dev/null
  iio_reg $PHY 0x035 $R35 >/dev/null 2>&1
  rxg 0 41; rxg 1 41
  i=0; while [ $i -le 7 ]; do
    [ -d $G/gpio$((OUT0+i)) ] && echo $((OUT0+i)) > $G/unexport 2>/dev/null; i=$((i+1)); done
  i=0; while [ $i -le 3 ]; do
    [ -d $G/gpio$((IN0+i)) ] && echo $((IN0+i)) > $G/unexport 2>/dev/null; i=$((i+1)); done
  echo "restored_0x0FB=$(r 0x0FB)"
  echo "restored_0x035=$(r 0x035)"
  echo "restored_tx2=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null|tail -1)"
  echo "restored_dds_raw=$(cat $DDS/out_altvoltage4_TX2_I_F1_raw)"
  echo "restored_idx=$(( $(r 0x2B0) & 0x7F )):$(( $(r 0x2B5) & 0x7F ))"
  S=0; i=0; while [ $i -le 7 ]; do [ -d $G/gpio$((OUT0+i)) ] && S=1; i=$((i+1)); done
  i=0; while [ $i -le 3 ]; do [ -d $G/gpio$((IN0+i)) ] && S=1; i=$((i+1)); done
  echo "all_pins_released=$( [ $S -eq 0 ] && echo true || echo false )"
}
trap restore EXIT INT TERM HUP

# CTRL_OUT as inputs
i=0; while [ $i -le 7 ]; do
  N=$((OUT0+i)); [ -d $G/gpio$N ] || echo $N > $G/export 2>/dev/null
  echo in > $G/gpio$N/direction 2>/dev/null; i=$((i+1)); done
# CTRL_IN as outputs, driven low and VERIFIED before arming
i=0; while [ $i -le 3 ]; do
  N=$((IN0+i)); [ -d $G/gpio$N ] || echo $N > $G/export 2>/dev/null
  echo out > $G/gpio$N/direction 2>/dev/null; echo 0 > $G/gpio$N/value 2>/dev/null; i=$((i+1)); done
PRE=0; i=0
while [ $i -le 3 ]; do
  D=$G/gpio$((IN0+i))
  [ "$(cat $D/direction)" = "out" ] || PRE=1
  [ "$(cat $D/value)" = "0" ] || PRE=1
  i=$((i+1))
done
echo "ctrl_in_verified_low=$( [ $PRE -eq 0 ] && echo true || echo false )"
[ $PRE -eq 0 ] || { echo "abort=ctrl_in_not_low"; exit 3; }

iio_reg $PHY 0x035 0x03 >/dev/null 2>&1
echo 1 > $DDS/out_altvoltage4_TX2_I_F1_raw
echo 1 > $DDS/out_altvoltage6_TX2_Q_F1_raw
txg 0; rxg 1 41
sleep 2

# ---------------- H4: latch and post-change blank ----------------
# drive RX1 well into large-ADC overload, then step gain DOWN one pin edge and sample
rxg 0 52
sleep 1
echo "h4_bits_in_overload=$(bits)"
echo "h4_idx_before=$(( $(r 0x2B0) & 0x7F ))"

# arm pin control (read-modify-write) so the gain can be stepped in ~134 us
ARM=$(( $(r 0x0FB) | 0x3 )); iio_reg $PHY 0x0FB $ARM >/dev/null 2>&1
echo "h4_armed=$(r 0x0FB)"

DEC=$G/gpio$((IN0+1))/value          # CTRL_IN1 = RX1 decrease (confirmed by H1)
OUT5=$G/gpio$((OUT0+5))/value        # CTRL_OUT5 = CH1 large ADC overload

# baseline rapid sample with NO gain change, to measure the sampler itself
s=""; n=0
while [ $n -lt $NSAMP ]; do read v < $OUT5; s="$s$v"; n=$((n+1)); done
echo "h4_control_no_change_trace=$s"

# now: pulse the decrease pin, then immediately sample
S=$(date +%s%N)
echo 1 > $DEC; echo 0 > $DEC
s=""; n=0
while [ $n -lt $NSAMP ]; do read v < $OUT5; s="$s$v"; n=$((n+1)); done
E=$(date +%s%N)
echo "h4_after_gain_step_trace=$s"
echo "h4_trace_total_us=$(( (E-S)/1000 ))"
echo "h4_us_per_sample=$(( (E-S)/1000/NSAMP ))"
echo "h4_idx_after=$(( $(r 0x2B0) & 0x7F ))"
echo "h4_bits_after=$(bits)"

# repeat a few times -- a blank shorter than one sample interval shows up as a
# sometimes-caught zero rather than never
k=1
while [ $k -le 5 ]; do
  rxg 0 52 >/dev/null 2>&1
  sleep 0.3
  echo 1 > $DEC; echo 0 > $DEC
  s=""; n=0
  while [ $n -lt 30 ]; do read v < $OUT5; s="$s$v"; n=$((n+1)); done
  echo "trial=h4_repeat:k=$k:trace=$s"
  k=$((k+1))
done

# disarm before leaving H4
V=$(r 0x0FB); iio_reg $PHY 0x0FB $(( $V & ~0x3 & 0xFF )) >/dev/null 2>&1
echo "h4_disarmed=$(r 0x0FB)"

# ---------------- H5: low-power bit update rate ----------------
# park just at the low-power threshold so the bit is marginal, then sample fast
OUT7=$G/gpio$((OUT0+7))/value
for g in 20 21 22 23 24; do
  rxg 0 $g; sleep 0.5
  S=$(date +%s%N); s=""; n=0
  while [ $n -lt 200 ]; do read v < $OUT7; s="$s$v"; n=$((n+1)); done
  E=$(date +%s%N)
  # count transitions in the trace
  echo "trial=h5:rx1_db=$g:us_per_sample=$(( (E-S)/1000/200 )):trace=$s"
done
echo "h4h5_complete=true"
exit 0

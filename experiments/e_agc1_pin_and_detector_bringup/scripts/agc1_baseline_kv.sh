#!/bin/sh
# E-AGC1 step 1 — baseline and safety. READ ONLY: no register writes, no GPIO exports.
# Emits flat key=value lines; the host assembles JSON and stamps the real time.
set -u
PHY=ad9361-phy
r() { iio_reg $PHY "$1" 2>/dev/null | tail -1; }

CHIP=""
for c in /sys/class/gpio/gpiochip*; do
  [ "$(cat "$c/label" 2>/dev/null)" = "zynq_gpio" ] && CHIP="$c"
done
BASE=$(cat "$CHIP/base"); NGPIO=$(cat "$CHIP/ngpio")

echo "gpio_chip=$(basename "$CHIP")"
echo "gpio_label=zynq_gpio"
echo "gpio_base=$BASE"
echo "gpio_ngpio=$NGPIO"
echo "ctrl_out0=$((BASE+54))"
echo "ctrl_out7=$((BASE+61))"
echo "ctrl_in0=$((BASE+62))"
echo "ctrl_in1=$((BASE+63))"
echo "ctrl_in2=$((BASE+64))"
echo "ctrl_in3=$((BASE+65))"
echo "en_agc=$((BASE+66))"
echo "resetb=$((BASE+67))"

echo "device_fw=$(awk '$1=="device-fw"{print $2}' /opt/VERSIONS)"
echo "uptime_s=$(cut -d' ' -f1 /proc/uptime)"
echo "kernel=$(uname -r)"

# every requested gpio line, so "nothing claims EMIO 0-11" is recorded not assumed
sed -n 's/^ *gpio-\([0-9]*\) *( *\([^)]*[^ )]\) *).*/claimed_line=\1:\2/p' \
    /sys/kernel/debug/gpio 2>/dev/null

echo "ensm_mode=$(iio_attr -d $PHY ensm_mode 2>/dev/null | tail -1)"
echo "ensm_mode_available=$(iio_attr -d $PHY ensm_mode_available 2>/dev/null | tail -1)"
echo "rx_lo_hz=$(iio_attr -c $PHY altvoltage0 frequency 2>/dev/null | tail -1)"
echo "tx_lo_hz=$(iio_attr -c $PHY altvoltage1 frequency 2>/dev/null | tail -1)"
echo "rx_sampling_hz=$(iio_attr -i -c $PHY voltage0 sampling_frequency 2>/dev/null | tail -1)"
echo "rx_rf_bandwidth_hz=$(iio_attr -i -c $PHY voltage0 rf_bandwidth 2>/dev/null | tail -1)"
echo "rx1_gain_control_mode=$(iio_attr -i -c $PHY voltage0 gain_control_mode 2>/dev/null | tail -1)"
echo "rx2_gain_control_mode=$(iio_attr -i -c $PHY voltage1 gain_control_mode 2>/dev/null | tail -1)"
echo "rx1_hardwaregain=$(iio_attr -i -c $PHY voltage0 hardwaregain 2>/dev/null | tail -1)"
echo "rx2_hardwaregain=$(iio_attr -i -c $PHY voltage1 hardwaregain 2>/dev/null | tail -1)"
echo "tx1_hardwaregain=$(iio_attr -o -c $PHY voltage0 hardwaregain 2>/dev/null | tail -1)"
echo "tx2_hardwaregain=$(iio_attr -o -c $PHY voltage1 hardwaregain 2>/dev/null | tail -1)"

for reg in 0x035 0x036 0x0FA 0x0FB 0x0FC 0x0FD 0x0FE 0x104 0x105 0x106 0x107 0x108 0x114 0x2B0 0x2B5; do
  echo "reg_$reg=$(r $reg)"
done

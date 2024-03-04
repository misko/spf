#!/bin/bash

if [ $# -ne 1 ]; then
        echo $0 ROVER_ID
        exit
fi

#assign the rover its ID
rover_id=$1
echo ${rover_id} > ~/rover_id

sudo apt-get update
sudo apt-get install git screen libiio-dev libiio-utils vim -y

# virtual enviornment setup
python -m venv ~/spf-virtualenv
source ~/spf-virtualenv/bin/activate
git clone https://github.com/misko/spf.git
cd spf
pip install -r requirements.txt

echo export PYTHONPATH=/home/pi/spf >> ~/.bashrc
echo 'test -z "$VIRTUAL_ENV" && source ~/spf-virtualenv/bin/activate' >> ~/.bashrc
echo "lsusb -t | grep usb-storage | sed 's/.*Port \([0-9]*\): Dev \([0-9]*\),.*/\1 \2/g' > ~/device_mapping" >> ~/.bashrc

# lets get eth0 to static ip
cat > interfaces <<- EOM
source /etc/network/interfaces.d/*

auto eth0
iface eth0 inet static
    address 192.168.1.__ROVERID__/24
    gateway 192.168.1.1
EOM
sed -i "s/__ROVERID__/$(expr 40 + ${rover_id})/g" interfaces
sudo cp -f interfaces /etc/network/interfaces

# add USB permissions for all users
sudo sh -c 'echo SUBSYSTEM=="usb_device", MODE="0664", GROUP="usb" >> /etc/udev/rules.d/99-com.rules'

# disable wifi so it wont interfere
sudo sh -c 'echo dtoverlay=disable-wifi >> /boot/config.txt'

#will come back up with new static IP
sudo reboot


# (spf-virtualenv) pi@roverpi2:~/spf/spf $ sudo iio_info -s
# Library version: 0.24 (git tag: v0.24)
# Compiled with backends: local xml ip usb
# Unable to create Local IIO context : No such file or directory (2)
# Available contexts:
#         0: 192.168.1.18 (Analog Devices PlutoSDR Rev.C (Z7010-AD9363A)), serial=1040007c4a94000211000b009186843ef2 [ip:pluto-2.local]
#         1: 192.168.1.17 (Analog Devices PlutoSDR Rev.C (Z7010-AD9363A)), serial=104000bac4950008230026001b440a003a [ip:pluto.local]
#         2: 192.168.2.1 (Analog Devices PlutoSDR Rev.C (Z7010-AD9363A)), serial=104000943807000a220011008101b882a9 [ip:pluto.local]
#         3: 0456:b673 (Analog Devices Inc. PlutoSDR (ADALM-PLUTO)), serial=104000943807000a220011008101b882a9 [usb:1.3.5]


# USB URI is usb:1.3.5

# #to allow users to access usb devices
# sudo chmod -R 777 /dev/bus/usb/
# # there is some examples online using udev but those are not owkring

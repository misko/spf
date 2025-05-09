#!/bin/bash

if [ $# -ne 1 ]; then
        echo $0 ROVER_ID
        exit
fi

#assign the rover its ID
repo_root="/home/pi/spf/"

rover_id=$1
echo ${rover_id} > ~/rover_id

cd /home/pi
git clone https://github.com/misko/spf.git

bash ${repo_root}/data_collection/rover/rover_v3.1/install_deps.sh

#sudo apt-get update
#sudo apt-get install git screen libiio-dev libiio-utils vim python3-dev uhubctl libusb-dev libusb-1.0-0-dev sshpass -y

# virtual enviornment setup
python -m venv ~/spf-virtualenv
source ~/spf-virtualenv/bin/activate

cd spf
#pip install -r requirements.txt
pip install -e . 
pip install RPi.GPIO

grep -v spf ~/.bashrc | grep -v lsusb  > /tmp/bashrc && mv /tmp/bashrc ~/.bashrc
echo export PYTHONPATH=/home/pi/spf >> ~/.bashrc
echo 'test -z "$VIRTUAL_ENV" && source ~/spf-virtualenv/bin/activate' >> ~/.bashrc

####PI 4####

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

# disable wifi so it wont interfere
grep -v disable-wifi /boot/config.txt > /tmp/config.txt && sudo cp /tmp/config.txt /boot/config.txt # /boot/firmware/config.txt (pi5)
sudo sh -c 'echo dtoverlay=disable-wifi >> /boot/config.txt' # /boot/firmware/config.txt pi5

### END PI 4 ####




####PI 5####
sudo apt-get install systemd-resolved
echo "lsusb  | grep PLUTO | awk '{print int(\$2)%3\" \"int(\$2)\" \"int(\$4) }' > ~/device_mapping" >> ~/.bashrc
# lets get eth0 to static ip
cat > interfaces <<- EOM
[Match]
Name=eth0

[Network]
Address=192.168.1.__ROVERID__/24
Gateway=192.168.1.1
DNS=8.8.8.8
EOM
sed -i "s/__ROVERID__/$(expr 40 + ${rover_id})/g" interfaces
sudo cp -f interfaces /etc/systemd/network/10-eth0.network
sudo systemctl enable systemd-networkd
sudo systemctl start systemd-networkd

# disable wifi so it wont interfere
grep -v disable-wifi /boot/firmware/config.txt > /tmp/config.txt && sudo cp /tmp/config.txt /boot/firmware/config.txt # /boot/firmware/config.txt (pi5)
sudo sh -c 'echo dtoverlay=disable-wifi >> /boot/firmware/config.txt' # pi5

#echo 8.8.8.8 | sudo tee /etc/resolv.conf 
sudo systemctl disable NetworkManager

### END PI 5 ####





# add USB permissions for all users
grep -v usb_device /etc/udev/rules.d/99-com.rules > /tmp/99-com.rules && sudo cp /tmp/99-com.rules /etc/udev/rules.d/99-com.rules
sudo sh -c 'echo SUBSYSTEM=="usb_device", MODE="0664", GROUP="usb" >> /etc/udev/rules.d/99-com.rules'

# flash pluto
# sudo chmod -R 777 /dev/bus/usb/ # allow all USB access

pluto_fw=plutosdr-fw-v0.37-dirty.zip

while [ 1 -lt 2 ]; do
        wget -O ${pluto_fw} 'https://www.dropbox.com/s/4jji77rk3d9ikba/plutosdr-fw-v0.37-dirty.zip?dl=0'
        md5='613fcdd4f45ad695d85abd53d1e0b918'
        current_md5=`md5sum ${pluto_fw} | awk '{print $1}'`
        if [ "$md5" == "${current_md5}" ]; then
                echo "Download succesful!"
                break
        fi
        rm -f ${pluto_fw}
done


mounts='/dev/sda /dev/sdb'
mount_point='/media/pluto'
if [ ! -d "${mount_point}" ]; then
        sudo mkdir -p ${mount_point}
fi
for mount in $mounts; do 
        if [ -b "$mount" ]; then
                echo "Trying to flash $mount"
                sudo mount "${mount}1" ${mount_point}
                sudo cp ${pluto_fw} ${mount_point}
                sudo eject ${mount}
                sleep 1
                echo -n "Waiting for device to come back up"
                while [ ! -b "${mount}1" ]; do
                        echo -n "."
                        sleep 5
                done
                echo "Device came back up!, flashing complete"
        fi
done


# flash rover
#sudo uhubctl  -a 0
#sleep 2
#sudo uhubctl  -a 1
#sleep 2
sudo systemctl stop mavlink_controller.service # might fail but just in case
sleep 5
wget https://raw.githubusercontent.com/ArduPilot/ardupilot/master/Tools/scripts/uploader.py
#wget https://firmware.ardupilot.org/Rover/stable-4.4.0/fmuv3/ardurover.apj
wget https://firmware.ardupilot.org/Rover/stable-4.5.0/fmuv3/ardurover.apj
#wget https://firmware.ardupilot.org/Rover/stable-4.4.0/fmuv2/ardurover.apj
python uploader.py ardurover.apj | tee > ardurover_flash.log
sleep 5
#cmd for mavproxy does not work
# go in by hand and do this a few times
#     param set FORMAT_VERSION 0
#     param show FORMAT_VERSION
#  then
#     reboot
# and restart mavproxy

#mavproxy.py  --cmd 'set requireexit True; param set FORMAT_VERSION 0; reboot; exit; exit;' # reset to default parameters
#sleep 5
#load these a few times, 
#should see HOLD> Loaded 256 parameters from this_rover.params (changed 0)


#mkfifo tmp_file
#cat tmp_file |  mavproxy.py &

#mavlink controller service
sudo cp /home/pi/spf/data_collection_model_and_results/rover/rover_v3.1/mavlink_controller.service /lib/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mavlink_controller.service

mkdir -p /home/pi/arduino
pushd /home/pi/arduino
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
popd
echo export PATH=${PATH}:/home/pi/arduino/bin >> ~/.bashrc
/home/pi/arduino/bin/arduino-cli config init
/home/pi/arduino/bin/arduino-cli config add board_manager.additional_urls https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json
/home/pi/arduino/bin/arduino-cli core update-index
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
# 
# # there is some examples online using udev but those are not owkring

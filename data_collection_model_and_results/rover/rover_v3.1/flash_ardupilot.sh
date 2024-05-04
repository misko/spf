sudo systemctl stop mavlink_controller.service # might fail but just in case
sleep 5
wget https://raw.githubusercontent.com/ArduPilot/ardupilot/master/Tools/scripts/uploader.py
#wget https://firmware.ardupilot.org/Rover/stable-4.4.0/fmuv3/ardurover.apj
wget https://firmware.ardupilot.org/Rover/stable-4.5.0/fmuv3/ardurover.apj
#wget https://firmware.ardupilot.org/Rover/stable-4.4.0/fmuv2/ardurover.apj
python uploader.py ardurover.apj | tee > ardurover_flash.log
sleep 5
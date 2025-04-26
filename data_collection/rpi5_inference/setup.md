repo_root=/home/pi/spf/

wget https://raw.githubusercontent.com/ArduPilot/ardupilot/master/Tools/scripts/uploader.py
wget https://firmware.ardupilot.org/Rover/stable-4.5.0/fmuv3/ardurover.apj
python uploader.py ardurover.apj | tee > ardurover_flash.log

cat base_params.params | sed "s/__ROVER_ID__/10/g" > this_drone.params
python ${repo_root}/spf/mavlink/mavlink_controller.py --load-params this_drone.params
python ${repo_root}/spf/mavlink/mavlink_controller.py --diff-params this_drone.params 
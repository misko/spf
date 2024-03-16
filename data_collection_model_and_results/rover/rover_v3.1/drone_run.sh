#!/bin/bash
repo_root=/home/pi/spf
export PYTHONPATH=${repo_root}
test -z "$VIRTUAL_ENV" && source ~/spf-virtualenv/bin/activate

rover_id=`cat /home/pi/rover_id`


echo "checking if updates available"
pushd ${repo_root}
current_hash=`git rev-parse --short HEAD`
git pull
new_hash=`git rev-parse --short HEAD`
if [ "${current_hash}" != "${new_hash}" ]; then
    echo "Detected git update going to try and reboot"
    echo "waiting for interrupt 15s..."
    sleep 15
    sudo cp ${repo_root}/data_collection_model_and_results/rover/rover_v3.1/mavlink_controller.service /lib/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable mavlink_controller.service
    sudo reboot
else
    echo "no updates (or maybe no internet) detected!"
fi
pip install -r requirements.txt
popd


rover_id=`cat ~/rover_id`

#make sure parameters are set correctly on ardupilot
params_root=${repo_root}/data_collection_model_and_results/rover/rover_v3.1/
cat ${params_root}/rover3_base_parameters.params ${params_root}/rover3_rc_servo_parameters.params | sed "s/__ROVER_ID__/${rover_id}/g" > this_rover.params
python ${repo_root}/spf/mavlink/mavlink_controller.py --diff-params this_rover.params --load-params this_rover.params
if [ $? -ne 0 ]; then
    echo "FAILED TO RESOLVE DIFFERENCES!!! running with incorrect params"
fi

#get GPS time
python ${repo_root}/spf/mavlink/mavlink_controller.py --get-time time
sudo date -s "$(cat time)"

echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

if [ ${rover_id} -eq 1 ]; then
    routine=bounce
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi.yaml 
elif [ ${rover_id} -eq 2 ]; then
    #config=${repo_root}/spf/rover_configs/rover_emitter_config_pi.yaml 
    config=${repo_root}/spf/rover_configs/rover_single_receiver_config_pi.yaml
    routine=circle
elif [ ${rover_id} -eq 3 ]; then
    routine=center
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi.yaml 
else
    echo Invalid rover_id 
    exit
fi

python3 ${repo_root}/spf/mavlink_radio_collection.py \
    -c ${config} -m /home/pi/device_mapping -r ${routine} -t "RO${rover_id}" -n 20000
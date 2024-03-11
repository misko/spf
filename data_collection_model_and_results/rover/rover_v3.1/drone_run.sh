#!/bin/bash
repo_root=/home/pi/spf
export PYTHONPATH=${repo_root}

rover_id=`cat /home/pi/rover_id`

echo "checking if updates available"
pushd /home/pi/spf/
current_hash=`git rev-parse --short HEAD`
git pull
new_hash=`git rev-parse --short HEAD`
if [ "${current_hash}" != "${new_hash}" ]; then
    echo "Detected git update going to try and reboot"
    echo "waiting for interrupt 15s..."
    sleep 15
    sudo cp /home/pi/spf/data_collection_model_and_results/rover/rover_v3.1/mavlink_controller.service /lib/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable mavlink_controller.service
fi
popd


if [ ${rover_id} -eq 1 ]; then
    routine=bounce
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi.yaml 
elif [ ${rover_id} -eq 2 ]; then
    config=${repo_root}/spf/rover_configs/rover_emitter_config_pi.yaml 
    routine=center
elif [ ${rover_id} -eq 3 ]; then
    routine=circle
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi.yaml 
else
    echo Invalid rover_id 
    exit
fi

/home/pi/spf-virtualenv/bin/python3 ${repo_root}/spf/mavlink_radio_collection.py \
    -c ${config} -m /home/pi/device_mapping -r ${routine} -t "RO${rover_id}"
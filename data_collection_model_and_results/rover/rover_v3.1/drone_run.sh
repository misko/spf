#!/bin/bash
repo_root=/home/pi/spf
export PYTHONPATH=${repo_root}

rover_id=`cat /home/pi/rover_id`

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

/home/pi/spf-virtualenv/bin/python3 ${repo_root}/spf/mavlink/mavlink_radio_collection.py \
    -c ${config} -m /home/pi/device_mapping -r ${routine}
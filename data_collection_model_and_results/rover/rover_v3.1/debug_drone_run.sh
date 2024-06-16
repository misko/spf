#!/bin/bash
repo_root=/home/pi/spf
export PYTHONPATH=${repo_root}
test -z "$VIRTUAL_ENV" && source ~/spf-virtualenv/bin/activate

rover_id=`cat /home/pi/rover_id`

if [ ! -f "/home/pi/.ssh/config" ]; then
	cp ${repo_root}/data_collection_model_and_results/rover/rover_v3.1/ssh_config /home/pi/.ssh/config
fi

rover_id=`cat ~/rover_id`

echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

if [ ${rover_id} -eq 1 ]; then
    routine=bounce
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi.yaml 
    n=200
elif [ ${rover_id} -eq 2 ]; then
    #config=${repo_root}/spf/rover_configs/rover_emitter_config_pi.yaml 
    config=${repo_root}/spf/rover_configs/rover_single_receiver_config_pi.yaml
    routine=circle
    n=200
elif [ ${rover_id} -eq 3 ]; then
    routine=center
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi.yaml 
    n=200
else
    echo Invalid rover_id 
    exit
fi

python3 ${repo_root}/spf/mavlink_radio_collection.py \
    -c ${config} -m /home/pi/device_mapping -r ${routine} -t "RO${rover_id}" -n $n --fake-drone --temp /dev/shm/

#python spf/spf/mavlink_radio_collection.py -c spf/spf/rover_configs/rover_receiver_config_pi.yaml  -m /home/pi/device_mapping -r  bounce -t "RO1" -n 40000 --drone-uri tcp:192.168.1.142:14591
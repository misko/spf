#!/bin/bash
repo_root=/home/pi/spf
export PYTHONPATH=${repo_root}
test -z "$VIRTUAL_ENV" && source ~/spf-virtualenv/bin/activate

rover_id=`cat /home/pi/rover_id`

if [ $# -eq 0 ]; then

	if [ ! -f "/home/pi/.ssh/config" ]; then
		cp ${repo_root}/data_collection_model_and_results/rover/rover_v3.1/ssh_config /home/pi/.ssh/config
	fi

	#check for internet
	sleep 10
	ping -c 1 8.8.8.8
	if [ $? -eq 0 ]; then
	    python ${repo_root}/spf/mavlink/mavlink_controller.py --buzzer git

	    echo "checking if updates available"
	    bash ${repo_root}/data_collection_model_and_results/rover/rover_v3.1/install_deps.sh
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
	    #pip install -r requirements.txt
	    pip install -e ${repo_root} 
	    popd
	fi
fi

rover_id=`cat ~/rover_id`


#make sure parameters are set correctly on ardupilot
params_root=${repo_root}/data_collection_model_and_results/rover/rover_v3.1/
cat ${params_root}/rover3_base_parameters.params ${params_root}/rover3_rc_servo_parameters.params | sed "s/__ROVER_ID__/${rover_id}/g" > this_rover.params
python ${repo_root}/spf/mavlink/mavlink_controller.py --load-params this_rover.params
python ${repo_root}/spf/mavlink/mavlink_controller.py --diff-params this_rover.params 
if [ $? -ne 0 ]; then
    echo "FAILED TO RESOLVE DIFFERENCES!!! running with incorrect params"
fi

#get GPS time
python ${repo_root}/spf/mavlink/mavlink_controller.py --get-time time
sudo date -s "$(cat time)"

echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

if [ ${rover_id} -eq 1 ]; then
    routine=diamond
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi_3mhz_35mm.yaml 
    n=3000
	expected_radios=2
elif [ ${rover_id} -eq 2 ]; then
    #config=${repo_root}/spf/rover_configs/rover_emitter_config_pi.yaml 
    config=${repo_root}/spf/rover_configs/rover_single_receiver_config_pi_3mhz.yaml
    routine=circle
    n=3500
	expected_radios=1
elif [ ${rover_id} -eq 3 ]; then
    routine=center
    config=${repo_root}/spf/rover_configs/rover_receiver_config_pi_3mhz.yaml 
    n=3000
	expected_radios=2
else
    echo Invalid rover_id 
    exit
fi

while [ 1 -gt 0 ]; do
	found_radios=`lsusb  | grep ADALM | wc -l`
	if [ ${found_radios} -eq ${expected_radios} ]; then
		break
	fi
	echo "EXPECTED ${expected_radios} but found ${found_radios}, trying again"
	python ${repo_root}/spf/mavlink/mavlink_controller.py --buzzer failure
	sleep 15
done


# do this a bit later to give them time to boot
echo "check pluto radios"
bash ${repo_root}/data_collection_model_and_results/rover/rover_v3.1/check_and_set_pluto.sh


if [ $# -eq 0 ]; then
	python3 ${repo_root}/spf/mavlink_radio_collection.py \
    	-c ${config} -m /home/pi/device_mapping -r ${routine} -t "RO${rover_id}" -n $n
else
	python3 ${repo_root}/spf/mavlink_radio_collection.py \
	  -c ${config}  -m /home/pi/device_mapping -r ${routine} -t "RO${rover_id}"  -n 40 --drone-uri tcp:192.168.1.141:14590 --no-ultrasonic
fi

#python spf/spf/mavlink_radio_collection.py -c spf/spf/rover_configs/rover_receiver_config_pi.yaml  -m /home/pi/device_mapping -r  bounce -t "RO1" -n 40000 --drone-uri tcp:192.168.1.136:14591
#python spf/spf/mavlink_radio_collection.py -c spf/spf/rover_configs/rover_receiver_config_pi.yaml  -m /home/pi/device_mapping -r  bounce -t "RO3" -n 40 --drone-uri tcp:192.168.1.136:14590 --no-ultrasonic

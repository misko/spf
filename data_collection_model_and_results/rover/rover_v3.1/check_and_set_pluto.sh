#!/bin/bash

eths=`ls /sys/class/net/ -l  | awk '{print $NF}' | grep usb | xargs -l basename`

function turn_on_eths {
	for ethx in ${eths}; do 
		sudo ifconfig ${ethx} up
	done
}

function turn_off_eths {
	for ethx in ${eths}; do 
		sudo ifconfig ${ethx} down
	done
}


function check_pluto {
	rm -f pluto_env
        sshpass -panalog ssh root@192.168.2.1 'fw_printenv' > pluto_env 2> /dev/null
	grep 'attr_name=compatible' pluto_env > /dev/null
	if [ $? -ne 0 ]; then
		return 1
	fi
	grep 'attr_val=ad9361' pluto_env > /dev/null
	if [ $? -ne 0 ]; then
		return 1
	fi
	grep 'compatible=ad9361' pluto_env > /dev/null
	if [ $? -ne 0 ]; then
		return 1
	fi
	return 0
}

function set_and_reboot_pluto {
	sshpass -panalog ssh root@192.168.2.1 'fw_setenv attr_name compatible; fw_setenv attr_val ad9361; fw_setenv compatible ad9361; fw_setenv mode 2r2t' > /dev/null 2>&1
        sshpass -panalog ssh root@192.168.2.1 'reboot' > /dev/null 2>&1
	

}

function wait_for_pluto {
	while [ 0 -eq 0 ]; do
		ssh-keygen -f "/home/pi/.ssh/known_hosts" -R "192.168.2.1"
		sshpass -panalog ssh -o ConnectTimeout=1 root@192.168.2.1 -o StrictHostKeyChecking=no uptime > /dev/null 2>&1 
		if [ $? -eq 0 ]; then
			return 0
		fi 
		sleep 0.5
	done
}

for eth in ${eths}; do
	turn_off_eths
	sudo ifconfig ${eth} up
	echo "Wait for pluto on $eth (might be usb eth)"
	wait_for_pluto
	check_pluto
	if [ $? -ne 0 ]; then
		echo "Setting fw param on $eth"
		set_and_reboot_pluto
		sleep 10 # wait for reboot to start
		echo "Waiting for pluto to come back up"
		wait_for_pluto
		check_pluto
		if [ $? -ne 0 ]; then
			echo "FAILED TO SET PARAMS ON $eth"
		else
			echo "Successfully set params on $eth"
		fi
	else
		echo "Params already set on $eth"
	fi
	wait_for_pluto # lets make sure its online before we turn off this interface again
done

turn_on_eths
sleep 0.5

lsusb -t | grep usb-storage | sed 's/.*Port \([0-9]*\): Dev \([0-9]*\),.*/\1 \2/g' > ~/device_mapping # update out USB mappings

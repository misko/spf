#!/bin/bash


function check_pluto {
	target_ip=$1
	rm -f pluto_env
        sshpass -panalog ssh root@${target_ip} 'fw_printenv' > pluto_env 2> /dev/null
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
	target_ip=$1
	sshpass -panalog ssh root@${target_ip} 'fw_setenv attr_name compatible; fw_setenv attr_val ad9361; fw_setenv compatible ad9361; fw_setenv mode 2r2t' > /dev/null 2>&1
        sshpass -panalog ssh root@${target_ip} 'reboot' > /dev/null 2>&1
	

}

function wait_for_pluto {
	target_ip=$1
	while [ 0 -eq 0 ]; do
		ssh-keygen -f ~/.ssh/known_hosts -R "${target_ip}"
		sshpass -panalog ssh -o ConnectTimeout=1 root@${target_ip} -o StrictHostKeyChecking=no uptime > /dev/null 2>&1 
		if [ $? -eq 0 ]; then
			return 0
		fi 
		sleep 0.5
	done
}

if [ $# -ne 1 ]; then
	echo $0 target_ip
	exit
fi

target_ip=$1
wait_for_pluto ${target_ip}
echo "Setting fw param on $eth"
set_and_reboot_pluto ${target_ip}
sleep 10 # wait for reboot to start
echo "Waiting for pluto to come back up"
wait_for_pluto ${target_ip}
check_pluto ${target_ip}
if [ $? -ne 0 ]; then
	echo "FAILED TO SET PARAMS ON $eth"
else
	echo "Successfully set params on $eth"
fi
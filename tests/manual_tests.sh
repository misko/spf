
#start docker simulator
#docker run --rm -it -p 14590-14595:14590-14595 ardupilot_spf /ardupilot/Tools/autotest/sim_vehicle.py \
#   -l 37.76509485,-122.40940127,0,0 -v rover -f rover-skid \
#    --out tcpin:0.0.0.0:14590  --out tcpin:0.0.0.0:14591 -S 5

repo_root=/Users/miskodzamba/Dropbox/research/gits/spf
py=/Users/miskodzamba/.virtualenvs/spf/bin/python

export PYTHONPATH=${repo_root}

function mavlink_controller() {
  $py ${repo_root}/spf/mavlink/mavlink_controller.py --ip 127.0.0.1 --port 14591 --proto tcp $@ > /dev/null 2>&1
}

function mavlink_radio_collection () {
  $py ${repo_root}/spf/mavlink_radio_collection.py -c ${repo_root}/tests/rover_config.yaml -m ${repo_root}/tests/device_mapping --fake-radio $@ > mavlink_radio_collection.log -n 500 2>&1 
}

echo "Test time since boot + reboot"  
rm -f time_since_boot1
mavlink_controller --time-since-boot time_since_boot1
if [ $? -ne 0 -o ! -f time_since_boot1 ]; then
  echo "Failed time since boot"
fi
start_time=$(date '+%s')
sleep 2
mavlink_controller --reboot
if [ $? -ne 0 ]; then
  echo "Failed reboot"
fi
rm -f time_since_boot2
mavlink_controller --time-since-boot time_since_boot2
if [ $? -ne 0 -o ! -f time_since_boot2 ]; then
  echo "Failed time since boot"
fi
end_time=$(date '+%s')
total_time=`expr ${end_time} - ${start_time}`
time_since_boot=`cat time_since_boot2`
if [ ${time_since_boot} -gt ${total_time} ]; then 
  echo "Failed reboot and time since boot test"
fi


echo "Test gps-time"
rm -f time
mavlink_controller --get-time time
if [ $? -ne 0 -o ! -f time ]; then
    echo "  Failed gps get time"
fi

echo "Test buzzer"
mavlink_controller --buzzer boot
if [ $? -ne 0  ]; then
    echo "  Failed run buzzer"
fi


echo "Test manual"
mavlink_controller --mode manual
if [ $? -ne 0  ]; then
    echo "  Failed manual mode"
fi


function check_dir () {
  tmpdir=$1
  ext=$2
  if [ ! -f ${tmpdir}/*.log${ext} ]; then
    echo "  Failed to find log file"
  fi
  if [ ! -f ${tmpdir}/*.npy${ext} ]; then
    echo "  Failed to find npy file"
  fi
  if [ ! -f ${tmpdir}/*.yaml${ext} ]; then
    echo "  Failed to find yaml file"
  fi
}


rover_id=5
params_root=${repo_root}/data_collection_model_and_results/rover/rover_v3.1/ 
cat ${params_root}/rover3_base_parameters.params | sed "s/__ROVER_ID__/${rover_id}/g" > ${repo_root}/tests/this_rover.params

echo "Test write SYSID 5"
mavlink_controller --load-params ${repo_root}/tests/this_rover.params
if [ $? -ne 0 ]; then
  echo "  Failed write SYSID 5"
fi

echo "Test diff with SYSID 5"
mavlink_controller --diff-params ${repo_root}/tests/this_rover.params
if [ $? -ne 0 ]; then
  echo "  Failed diff SYSID 5"
fi

rover_id=6
params_root=${repo_root}/data_collection_model_and_results/rover/rover_v3.1/ 
cat ${params_root}/rover3_base_parameters.params | sed "s/__ROVER_ID__/${rover_id}/g" > ${repo_root}/tests/this_rover.params

echo "Test diff with SYSID 6"
mavlink_controller --diff-params ${repo_root}/tests/this_rover.params
if [ $? -eq 0 ]; then
  echo "  Failed diff SYSID 6"
fi

echo "Test write with SYSID 6"
mavlink_controller --load-params ${repo_root}/tests/this_rover.params
if [ $? -ne 0 ]; then
  echo "  Failed write SYSID 6"
fi

mavlink_controller --diff-params ${repo_root}/tests/this_rover.params
echo "Test diff with SYSID 6"
if [ $? -ne 0 ]; then
  echo "  Failed diff SYSID 6"
fi

echo "Test record radio manual mode"
tmpdir=`mktemp -d`
mavlink_radio_collection -r circle --temp ${tmpdir} -s 10
if [ $? -ne 0 ]; then
  echo "  Failed record radio manual mode"
fi
grep "MavRadioCollection: Waiting for drone to start moving" mavlink_radio_collection.log > /dev/null
if [ $? -ne 0 ]; then 
  echo "  Failed record radio manual mode - start moving"
fi
grep "Planner starting to issue move commands" mavlink_radio_collection.log > /dev/null
if [ $? -eq 0 ]; then 
  echo "  Failed record radio manual mode - start moving"
fi
check_dir $tmpdir ".tmp"
rm ${tmpdir}/*
rmdir $tmpdir

echo "Test fakemode"
mavlink_controller --mode fakemode
if [ $? -ne 1 ]; then
    echo "Failed fakemode mode"
fi

echo "Test guided mode"
mavlink_controller --mode guided
if [ $? -ne 0  ]; then
    echo "Failed guided mode"
fi

echo "Test record radio guided mode"
tmpdir=`mktemp -d`
mavlink_radio_collection -r circle --temp ${tmpdir} -s 30
if [ $? -ne 0 ]; then
  echo "  Failed record radio manual mode"
fi
grep "Planner starting to issue move commands" mavlink_radio_collection.log > /dev/null
if [ $? -ne 0 ]; then 
  echo "  Failed record radio manual mode - start moving"
fi
check_dir $tmpdir ""
rm ${tmpdir}/*
rmdir $tmpdir


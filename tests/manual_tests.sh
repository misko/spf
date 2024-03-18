
#start docker simulator
#docker run --rm -it -p 14590-14595:14590-14595 ardupilot_spf /ardupilot/Tools/autotest/sim_vehicle.py \
#   -l 37.76509485,-122.40940127,0,0 -v rover -f rover-skid \
#    --out tcpin:0.0.0.0:14590  --out tcpin:0.0.0.0:14591

repo_root=/Users/miskodzamba/Dropbox/research/gits/spf
py=/Users/miskodzamba/.virtualenvs/spf/bin/python

export PYTHONPATH=${repo_root}

function mavlink_controller() {
  $py ${repo_root}/spf/mavlink/mavlink_controller.py --ip 127.0.0.1 --port 14591 --proto tcp $@ > /dev/null 2>&1
}

function mavlink_radio_collection () {
  $py ${repo_root}/spf/mavlink_radio_collection.py -c ${repo_root}/tests/rover_config.yaml -m ${repo_root}/tests/device_mapping --fake-radio $@ > mavlink_radio_collection.log 2>&1 
}

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
  if [ ! -f ${tmpdir}/*.log.tmp ]; then
    echo "  Failed to find log file"
  fi
  if [ ! -f ${tmpdir}/*.npy.tmp ]; then
    echo "  Failed to find npy file"
  fi
  if [ ! -f ${tmpdir}/*.yaml.tmp ]; then
    echo "  Failed to find yaml file"
  fi
}

echo "Test record radio manual mode"
tmpdir=`mktemp -d`
mavlink_radio_collection -r circle --temp ${tmpdir} -s 10
if [ $? -ne 0 ]; then
  echo "  Failed record radio manual mode"
fi
check_dir $tmpdir 
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
echo 
check_dir $tmpdir
rm ${tmpdir}/*
rmdir $tmpdir





repo_root=/Users/miskodzamba/Dropbox/research/gits/spf
py=/Users/miskodzamba/.virtualenvs/spf/bin/python

export PYTHONPATH=${repo_root}

#start docker simulator
#docker run --rm -it -p 14590-14595:14590-14595 ardupilot_spf /ardupilot/Tools/autotest/sim_vehicle.py \
#   -l 37.76509485,-122.40940127,0,0 -v rover -f rover-skid \
#    --out tcpin:0.0.0.0:14590  --out tcpin:0.0.0.0:14591

function mavlink_controller() {
  $py ${repo_root}/spf/mavlink/mavlink_controller.py --ip 127.0.0.1 --port 14591 --proto tcp $@ > /dev/null 2>&1
}

echo "Test gps-time"
rm -f time
mavlink_controller --get-time time
if [ $? -ne 0 -o ! -f time ]; then
    echo "Failed gps get time"
fi

echo "Test buzzer"
mavlink_controller --buzzer boot
if [ $? -ne 0  ]; then
    echo "Failed run buzzer"
fi

echo "Test guided mode"
mavlink_controller --mode guided
if [ $? -ne 0  ]; then
    echo "Failed guided mode"
fi

echo "Test manual"
mavlink_controller --mode manual
if [ $? -ne 0  ]; then
    echo "Failed manual mode"
fi

echo "Test fakemode"
mavlink_controller --mode fakemode
if [ $? -ne 1 ]; then
    echo "Failed fakemode mode"
fi

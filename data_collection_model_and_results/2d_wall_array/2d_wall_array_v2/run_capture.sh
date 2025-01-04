pluto_ips="192.168.1.17 192.168.1.18"

reboot_plutos () {
    for pluto_ip in ${pluto_ips}; do
        bash set_and_reboot_pluto.sh ${pluto_ip}
    done
}

apcaccess | grep STATUS | grep ONBATT 
if [ $? -eq 0 ]; then
       echo REFUSING TO RUN SCRIPT WHILE ON BATTERY
       exit
fi       

export ROOT="../../../"
#config=${ROOT}/spf/v5_configs/wall_array_v2_external_70_max_5MhzBandwidth_manual.yaml
#config=${ROOT}/spf/v5_configs/wall_array_v2_external_70_max_5MhzBandwidth.yaml
config=${ROOT}/spf/v5_configs/wall_array_v2_external_70_max_5MhzBandwidth_manual60.yaml

#reboot_plutos
sleep 3
python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --dry-run --n-records 1
sleep 1

python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r rx_random_circle -s /dev/ttyACM0 --n-records 5000
sleep 1

python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --dry-run --n-records 1

reboot_plutos 
python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --dry-run --n-records 1

python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --n-records 10000

reboot_plutos
python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --dry-run --n-records 1

python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --n-records 10000

#python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r rx_random_circle -s /dev/ttyACM0 --n-records 10000
sleep 1

reboot_plutos
python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --dry-run --n-records 1

python ${ROOT}/spf/grbl_radio_collection.py  -c ${config} -r bounce -s /dev/ttyACM0 --n-records 10000

cat ./go_home | python ${ROOT}/spf/grbl/grbl_interactive.py /dev/ttyACM0

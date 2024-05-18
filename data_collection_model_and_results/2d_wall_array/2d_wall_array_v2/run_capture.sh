pluto_ips="192.168.1.17 192.168.1.18"

reboot_plutos () {
    for pluto_ip in ${pluto_ips}; do
        bash set_and_reboot_pluto.sh ${pluto_ip}
    done
}

export ROOT="../../../"

reboot_plutos
python ${ROOT}/spf/grbl_radio_collection.py  -c ${ROOT}/spf/v5_configs/wall_array_v2_external.yaml -r rx_circle -s /dev/ttyACM0
reboot_plutos
python ${ROOT}/spf/grbl_radio_collection.py  -c ${ROOT}/spf/v5_configs/wall_array_v2_external.yaml -r bounce -s /dev/ttyACM0
reboot_plutos
python ${ROOT}/spf/grbl_radio_collection.py  -c ${ROOT}/spf/v5_configs/wall_array_v2_external.yaml -r bounce -s /dev/ttyACM0
reboot_plutos
python ${ROOT}/spf/grbl_radio_collection.py  -c ${ROOT}/spf/v5_configs/wall_array_v2_external.yaml -r bounce -s /dev/ttyACM0
cat ./go_home | python ${ROOT}/spf/grbl/grbl_interactive.py /dev/ttyACM0

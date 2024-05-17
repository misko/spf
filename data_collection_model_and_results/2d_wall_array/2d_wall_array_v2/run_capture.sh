pluto_ips="192.168.1.17 192.168.1.18"

reboot_plutos () {
    for pluto_ip in ${pluto_ips}; do
        bash set_and_reboot_pluto.sh ${pluto_ip}
    done
}

reboot_plutos
python spf/grbl_radio_collection.py  -c spf/v5_configs/wall_array_v2_external.yaml -r rx_circle -s /dev/ttyACM0
reboot_plutos
python spf/grbl_radio_collection.py  -c spf/v5_configs/wall_array_v2_external.yaml -r bounce -s /dev/ttyACM0
reboot_plutos
python spf/grbl_radio_collection.py  -c spf/v5_configs/wall_array_v2_external.yaml -r bounce -s /dev/ttyACM0
reboot_plutos
python spf/grbl_radio_collection.py  -c spf/v5_configs/wall_array_v2_external.yaml -r bounce -s /dev/ttyACM0
cat spf/grbl/snippets/go_home | python spf/grbl/grbl_interactive.py /dev/ttyACM0
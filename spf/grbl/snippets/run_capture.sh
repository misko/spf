python spf/grbl_radio_collection.py  -c spf/v5_configs/wall_array_v2_external.yaml -r rx_circle -s /dev/ttyACM0
python spf/grbl_radio_collection.py  -c spf/v5_configs/wall_array_v2_external.yaml -r bounce -s /dev/ttyACM0
cat spf/grbl/snippets/go_home | python spf/grbl/grbl_interactive.py /dev/ttyACM0
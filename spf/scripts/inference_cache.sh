#!/bin/bash
if [ $# -ne 1 ]; then
	echo $0 checkpoint_dir
	exit
fi
dir=$1
precompute_cache=/mnt/4tb_ssd/precompute_cache_new
input_zarrs=/mnt/4tb_ssd/nosig_data/wallarrayv3_2024_*.zarr
root=/home/mouse9911/gits/spf/
inference_cache=/mnt/4tb_ssd/inference_cache/ 
python ${root}/spf/scripts/create_inference_cache.py --inference-cache ${inference_cache} --config-fn ${dir}/config.yml --checkpoint-fn ${dir}/best.pth --device cuda --datasets ${input_zarrs} --parallel 24 --precompute-cache ${precompute_cache} 

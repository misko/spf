#!/bin/bash
# if [ $# -ne 1 ]; then
# 	echo $0 checkpoint_dir
# 	exit
# fi
#dir=$1

root=/home/mouse9911/gits/spf/
inference_cache=/mnt/4tb_ssd/inference_cache_nov29/ 
input_zarrs=/mnt/4tb_ssd/nosig_data/wallarrayv3_2024_*.zarr
input_zarrs=`cat  /mnt/4tb_ssd/nosig_data/train_val_fullpath.txt`

# 3.2 
checkpoints="/home/mouse9911/gits/spf/nov22_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_beamformerOnly_withMag /home/mouse9911/gits/spf/nov26_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05 /home/mouse9911/gits/spf/nov28_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun"
precompute_cache=/mnt/4tb_ssd/precompute_cache_new
segmentation_version=3.2

# 3.3 
# precompute_cache=/mnt/4tb_ssd/precompute_cache_3p3
# segmentation_version=3.3

#3.11
# checkpoints="/home/mouse9911/gits/spf/nov28_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun_3p11 /home/mouse9911/gits/spf/nov28_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun"
# precompute_cache=/mnt/4tb_ssd/precompute_cache
# segmentation_version=3.11

for dir in $checkpoints; do
python ${root}/spf/scripts/create_inference_cache.py --inference-cache ${inference_cache} --config-fn ${dir}/config.yml --checkpoint-fn ${dir}/best.pth --device cuda --datasets ${input_zarrs} --parallel 12	 --precompute-cache ${precompute_cache} --segmentation-version ${segmentation_version} #--debug
done
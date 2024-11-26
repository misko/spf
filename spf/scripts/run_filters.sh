#!/bin/bash
val_file=/mnt/4tb_ssd/nosig_data/nov23_val.txt
root=/home/mouse9911/gits/spf/
precompute_cache=/mnt/4tb_ssd/precompute_cache_new
input_zarrs=/mnt/4tb_ssd/nosig_data/wallarrayv3_2024_*.zarr
root=/home/mouse9911/gits/spf
inference_cache=/mnt/4tb_ssd/inference_cache/ 

cat ${val_file} | while read x; do 
   python ${root}/spf/scripts/run_filters_on_data.py -d $x --nthetas 65 --device cpu --skip-qc  \
	   --precompute-cache ${precompute_cache} --empirical-pkl-fn ${root}/empirical_dists/full.pkl \
	   --parallel 24 --work-dir ${root}/spf/run_on_filters_nov25 --config ${root}/spf/model_training_and_inference/models/ekf_and_pf_config.yml
done

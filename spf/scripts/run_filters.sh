#!/bin/bash
#val_file=/mnt/4tb_ssd/nosig_data/nov23_val.txt



val_file=/mnt/4tb_ssd/nosig_data/val.txt
root=/home/mouse9911/gits/spf

shuf --random-source ${val_file} ${val_file} | while read x; do 
   echo python ${root}/spf/scripts/run_filters_on_data.py -d $x --nthetas 65 --device cpu --skip-qc  \
	--empirical-pkl-fn ${root}/empirical_dists/full.pkl \
	--parallel 24  --work-dir ${root}/spf/run_on_filters_nov30 --config ${root}/spf/model_training_and_inference/models/ekf_and_pf_config.yml
done

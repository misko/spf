torun="/home/mouse9911/gits/spf/dec3_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun_3p4 \
/home/mouse9911/gits/spf/dec3_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun_3p4_windows_windowedbeamformer_nophase_nobeam_noemp_normalized \
/home/mouse9911/gits/spf/dec3_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun_3p4_windows_windowedbeamformer_nophase_nobeam_noemp_normalized_big2 \
/home/mouse9911/gits/spf/dec3_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun_3p4_windows_fix_rerun"
for checkpoint_dir in $torun; do 
    config=${checkpoint_dir}/config.yml
    checkpoint=${checkpoint_dir}/best.pth
    echo ${config_and_checkpoint}
    python create_inference_cache.py --config-fn ${config} --checkpoint-fn ${checkpoint} --inference-cache /mnt/md2/cache/inference --segmentation-version 3.4 --precompute-cache /mnt/md2/cache/precompute_cache_3p4 -d /mnt/4tb_ssd/nosig_data/train.txt 
    python create_inference_cache.py --config-fn ${config} --checkpoint-fn ${checkpoint} --inference-cache /mnt/md2/cache/inference --segmentation-version 3.4 --precompute-cache /mnt/md2/cache/precompute_cache_3p4 -d /mnt/4tb_ssd/nosig_data/val.txt  
done
#/home/mouse9911/gits/spf/latest_configs/dec4_paired_sigma0p05_rerun_3p4_windows_fix.yaml,/home/mouse9911/gits/spf/dec3_checkpoints/paired_checkpoints_inputdo0p3_sigma0p05_rerun_3p4_windows_fix \
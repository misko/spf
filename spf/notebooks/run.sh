stepss="480000" #"60000 480000"
lrs="0.000001" # 0.0000005 0.000002"
tmodels="512" # 1024"
thids="1024" # 2048"
snapshots="8" #"1 8 32"
layers="16" #"8 16 32 64"

for steps in $stepss; do
for tmodel in $tmodels; do
for thid in $thids; do
for layer in $layers; do
for lr in $lrs; do
for snapshot in $snapshots; do
step=`expr $steps / $snapshot`
python simple_train_filter.py -d /mnt/md0/spf/2d_wallarray_v2_data/june_fix/wallarrayv3_2024_06*e.zarr \
	--precompute-cache ~/precompute_cache_chunk16_fresh/ --act leaky --skip-qc --workers 8 \
	--batch 128 --device cuda --wandb-project projectspf_aug13_clean3 --lr $lr --val-holdout-fraction 0.3 \
	--val-subsample-fraction 1.0  --tformer-dmodel $tmodel --tformer-dhid $thid --snapshots-per-session $snapshot \
       	--head-start 0 --amp --snapshots-stride 1 --load-beamnet beamnet.chkpnt --tformer-dropout 0.5 \
	--tformer-snapshot-dropout 0.5 --tformer-layers $layer --steps $step
done
done
done
done
done
done

stepss="32000000" #"60000 480000"
lrs="0.000001 0.0000001" #0.00000001 0.0000001 0.000001" # 0.0000005 0.000002"
layers="16" # 32" #"8 16 32 64"
hiddens="128 2048 512" #"8 16 32 64"
batches="256 2048 512"
wds="0.0001"
norms="batch layer"

for steps in $stepss; do
for hid in $hiddens; do
for layer in $layers; do
for lr in $lrs; do
for batch in $batches; do
for wd in $wds; do 
for norm in $norms; do
step=`expr $steps / $batch`
val_every=`expr $step / 18`
head_start=`expr $step / 2`
python simple_train_filter.py -d /mnt/md0/spf/2d_wallarray_v2_data/june_fix/*.zarr --precompute-cache ~/precompute_cache_chunk16_fresh/ \
	--act leaky --skip-qc --workers 5 --batch $batch --device cuda --wandb-project projectspf_sep5_direct --lr $lr \
	--val-holdout-fraction 0.08 --val-subsample-fraction 1.0  --tformer-dmodel 128 --tformer-dhid 128 --val-every ${val_every} \
	--snapshots-per-session 1 --head-start ${head_start} --amp --snapshots-stride 1  --tformer-dropout 0.0 --tformer-snapshot-dropout 0.0 \
	--tformer-layers 2 --beam-net-hidden $hid --beam-net-depth $layer --steps $step --weight-decay $wd --beam-norm-type $norm
done
done
done
done
done
done
done
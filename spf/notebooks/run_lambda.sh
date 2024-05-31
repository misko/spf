# python simple_train.py -d /home/ubuntu/data/wallarrayv3_2024_05_17_17_50_39_nRX2_rx_circle.zarr /home/ubuntu/data/wallarrayv3_2024_05_18_00_55_58_nRX2_bounce.zarr /home/ubuntu/data/wallarrayv3_2024_05_18_07_53_17_nRX2_bounce.zarr /home/ubuntu/data/wallarrayv3_2024_05_18_16_29_28_nRX2_rx_circle.zarr /home/ubuntu/data/wallarrayv3_2024_05_18_23_32_12_nRX2_bounce.zarr 
# --device cuda --batch 64 --workers 28 --type direct 
# --lr 0.001 --shuffle --segmentation-level downsampled 
# --epochs 3 --depth 5 --hidden 128 --act leaky --other  --seg-net conv
#  --skip-segmentation --wandb-project may29run_WTF --batch-norm --block 


bns="x bn"
acts="leaky selu"
lrs="0.001 0.0001 0.01"
wd="0.0 0.00001 0.0001"
types="direct"
others="other" # x"
symmetrys="symmetry" # x"
circles="x" # circular-mean"
sigs="x" # no-sigmoid"
blocks="x" # block"
depths="5" # 8"
hiddens="64" # 128"
head_starts="0 1000 4000"
seg_nets="conv" # "conv"
for type in $types; do 
for lr in $lrs; do 
for seg_net in ${seg_nets}; do
for depth in $depths; do
for hidden in $hiddens; do
for bn in $bns; do
for symmetry in $symmetrys; do 
for act in $acts; do 
for other in $others; do
for circular in $circles; do
for sig in ${sigs}; do
for block in $blocks; do
#for start in ${head_starts}; do
bn_flag="--batch-norm" 
if [ $bn ==  "x" ]; then
bn_flag=""
fi
sym="--symmetry"
if [ "$symmetry" == "x" ]; then
  sym=""
fi
oth="--other"
if [ "$other" == "x" ]; then
  oth=""
fi
if [ $depth -ne 2 -a "${seg_net}" == "unet" ]; then
  continue
fi
sigf="--sigmoid"
if [ "$sig" == "x" ]; then
  sigf=""
fi
cir="--circular-mean"
if [ "$circular" == "x" ]; then
  cir=""
fi
blk="--block"
if [ "$block" == "x" ]; then
  blk=""
fi
echo python simple_train.py  -d ~/data/*.zarr  --device cuda \
 --batch 128 --workers 16 --type $type --lr $lr --shuffle \
 --segmentation-level downsampled --epochs 8 --depth ${depth} \
 --hidden ${hidden} --act $act $oth $sym $cir $sigf $blk \
 --seg-net ${seg_net} ${bn_flag} --wandb-project may30v1 #--seg-start ${start}
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
 done
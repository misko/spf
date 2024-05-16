bns="x bn"
acts="selu leaky"
lrs="0.001"
types="direct"
others="x other"
symmetrys="x symmetry"
depths="2 5 8"
hiddens="16 32 64"
head_starts="0 1000 4000"
seg_nets="unet conv"
for bn in $bns; do
for symmetry in $symmetrys; do 
for act in $acts; do 
for type in $types; do 
for lr in $lrs; do 
for other in $others; do
for depth in $depths; do
for hidden in $hiddens; do
for seg_net in ${seg_nets}; do
for start in ${head_starts}; do
bn_flag="--batch-norm" 
if [ $bn ==  "x" ]; then
bn_flag=""
fi
sym="--symmetry"
if [ "$symmetry" == "x" ]; then
  sym=""
fi
other="--other"
if [ "$other" == "x" ]; then
  other=""
fi
if [ $depth -ne 2 -a "${seg_net}" == "unet" ]; then
  continue
fi
python simple_train.py  -d ~/data/*.zarr  --device cuda \
 --batch 64 --workers 28 --type $type --lr $lr \
 --segmentation-level downsampled --epochs 5 --depth ${depth} \
 --hidden ${hidden} --act $act $other $sym --seg-start ${start} \
 --seg-net ${seg_net} ${bn_flag} --skip-segmentation
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
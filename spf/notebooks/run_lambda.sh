bns="x bn"
acts="selu leaky"
lrs="0.001 0.0001 0.01"
types="direct"
others="x other"
symmetrys="x symmetry"
depths="5 8"
hiddens="64 128"
head_starts="0 1000 4000"
seg_nets="unet conv"
for type in $types; do 
for lr in $lrs; do 
for seg_net in ${seg_nets}; do
for depth in $depths; do
for hidden in $hiddens; do
for bn in $bns; do
for symmetry in $symmetrys; do 
for act in $acts; do 
for other in $others; do
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
python simple_train.py  -d ~/data/*.zarr  --device cuda \
 --batch 64 --workers 28 --type $type --lr $lr \
 --segmentation-level downsampled --epochs 3 --depth ${depth} \
 --hidden ${hidden} --act $act $oth $sym  \
 --seg-net ${seg_net} ${bn_flag} --skip-segmentation #--seg-start ${start}
 done
 done
 done
 done
 done
 done
 done
 done
 done
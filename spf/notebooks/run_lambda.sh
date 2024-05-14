acts="selu leaky"
lrs="0.001"
types="direct"
others="x other"
symmetrys="x symmetry"
depths="2 5 8"
hiddens="16 32 64"
head_starts="0 1000 4000"
for act in $acts; do 
for type in $types; do 
for lr in $lrs; do 
for other in $others; do
for symmetry in $symmetrys; do 
for depth in $depths; do
for hidden in $hiddens; do
for start in ${head_starts}; do
sym="--symmetry"
if [ "$symmetry" == "x" ]; then
  sym=""
fi
other="--other"
if [ "$other" == "x" ]; then
  other=""
fi
python simple_train.py  -d ~/data/*.zarr  --device cuda \
 --batch 64 --workers 28 --type $type --lr $lr \
 --segmentation-level downsampled --epochs 20 --depth ${depth} \
 --hidden ${hidden} --act $act $other $sym --seg-start ${start}
 done
 done
 done
 done
 done
 done
 done
 done
 done
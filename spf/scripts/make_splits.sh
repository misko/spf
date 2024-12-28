
splits=/mnt/md2/splits/
name=dec26
n=90 # 90% for train

ls /mnt/ssd/2d_wallarray_v2_data/*/*.zarr -d | shuf > ${splits}/${name}_full.txt
grep rx_circle ${splits}/${name}_full.txt > ${splits}/${name}_val.txt
grep -v rx_circle ${splits}/${name}_full.txt > ${splits}/${name}_notcircle.txt

total=$(wc -l < "${splits}/${name}_notcircle.txt")
cutoff=$(( (n * total) / 100 ))

head -n "$cutoff" "${splits}/${name}_notcircle.txt" > ${splits}/${name}_train.txt
tail -n $(( total - cutoff )) "${splits}/${name}_notcircle.txt" >> ${splits}/${name}_val.txt

# add in rover to validation
ls /mnt/ssd/rovers/merged/nov*.zarr -d >> ${splits}/${name}_train.txt
ls /mnt/ssd/rovers/merged/dec*.zarr -d | grep -v dec26 >> ${splits}/${name}_train.txt
ls /mnt/ssd/rovers/merged/dec*.zarr -d | grep dec26 >> ${splits}/${name}_val.txt

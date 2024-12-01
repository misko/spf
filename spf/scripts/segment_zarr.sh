#!/bin/bash


# for x in /mnt/md1/2d_wallarray_v2_data/*/*.zarr /mnt/md0/spf/2d_wallarray_v2_data/oct_batch2/*.zarr; do python spf/scripts/segment_zarr.py --input-zarr $x --precompute-cache /mnt/4tb_ssd/precompute_cache_3p3/ --gpu -p 16; done

if [ $# -ne 1 ] ; then
	echo $0 input_file
	exit
fi

cat $1 | while read line; do 
	echo "processing $line"
	fadvise -a willneeded $line 
	python segment_zarr.py --input-zarr $line --precompute-cache /mnt/4tb_ssd/precompute_cache_new/ --gpu -p 16
	fadvise -a dontneed $line 
done

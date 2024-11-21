#!/bin/bash

if [ $# -ne 1 ] ; then
	echo $0 input_file
	exit
fi

cat $1 | while read line; do 
	echo "processing $line"
	fadvise -a willneeded $line 
	python segment_zarr.py --input-zarr $line --precompute-cache /mnt/4tb_ssd/precompute_cache_new/ --gpu -p 12
	fadvise -a dontneed $line 
done

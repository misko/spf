#!/bin/bash

if [ $# -ne 2 ]; then
	echo $0 target_link input
	exit
fi

target=$1
input=$2
input_basename="${input%.*}"

for x in ${input_basename}.*; do 
	ext="${x##*.}"
	ln -s $x ${target}.${ext} 
done

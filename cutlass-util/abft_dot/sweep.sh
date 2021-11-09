#!/bin/bash

NUM_ARGS=6
if [ $# -ne $NUM_ARGS ]; then
  echo "Usage: $0 <precision> <M> <N> <K> <num_rows_A> <partial_output_size>"
  exit 1
fi

prec=$1
M=$2
N=$3
K=$4
num_rows_A=$5
partial_output_size=$6

if [ "$prec" != "fp32" ] && [ "$prec" != "fp16" ]; then
  echo "<precision> must be one of {fp32, fp16}. Got ${prec}"
  exit 1
fi

exe=dot_${prec}.out
if [ ! -f $exe ]; then
  make $exe
fi

tmp_file=/tmp/__dotsweep.txt
echo "split_m,block_size,time" > $tmp_file
min_time=10000000
min_block_size=0
min_split_m=0
iterations=100
for split_m in 1 2 4 8 16 32 64 128 256 512 1024 2048; do
  if [ $split_m -gt $num_rows_A ]; then
    continue
  fi

  for single_kernel in 0 1; do
    if [ $single_kernel -eq 1 ] && [ $split_m -ne 1 ]; then
      continue
    fi

    for block_size in 512 256 128 64 32; do
      time=$(./$exe $M $N $K ${num_rows_A} ${partial_output_size} ${block_size} ${split_m} ${single_kernel} 0 $iterations | grep "Time" | awk -F'=' '{print $NF}' | awk -F'ms' '{print $1}')
      echo "${split_m},${block_size},${time}" >> $tmp_file

      if (( $(echo "$time < $min_time" | bc -l) )); then
        min_time=$time
        min_block_size=$block_size
        min_split_m=$split_m
        min_single_kernel=$single_kernel
      fi
    done
  done
done

echo $min_time
#echo "block_size = $min_block_size"
#echo "split_m = $min_split_m"
#echo "single_kernel = $min_single_kernel"

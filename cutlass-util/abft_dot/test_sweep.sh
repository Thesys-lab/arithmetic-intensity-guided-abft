#!/bin/bash

NUM_ARGS=1
if [ $# -ne $NUM_ARGS ]; then
  echo "Usage: $0 <precision>"
  exit 1
fi

prec=$1
num_rows_A=8
partial_output_size=64

if [ "$prec" != "fp32" ] && [ "$prec" != "fp16" ]; then
  echo "<precision> must be one of {fp32, fp16}. Got ${prec}"
  exit 1
fi

exe=dot_${prec}.out
if [ ! -f $exe ]; then
  make $exe
fi

gemmdir=$CUTLASS_UTIL_DIR/gemm/gemm_files
for fil in $(ls $gemmdir/*1080*.txt) $(ls $gemmdir/*224*.txt); do
  fil=$(echo $fil | awk -F'/' '{print $NF}')
  model=$(echo $fil | awk -F'_' '{print $1}')
  res=$(echo $fil | awk -F'_' '{print $NF}' | awk -F'.txt' '{print $1}')
  fullfil=$gemmdir/$fil
  echo "================== $fil =================="
  while IFS=' ' read -r M N K count; do
    echo "$M $N $K"
    for split_m in 1; do # 2 4 8 16 32 64 128 256 512 1024 2048; do
      if [ $split_m -gt $num_rows_A ]; then
        continue
      fi

      for single_kernel in 0 1; do
        if [ $single_kernel -eq 1 ] && [ $split_m -ne 1 ]; then
          continue
        fi
        for block_size in 512 256 128 64 32; do
          result=$(./$exe $M $N $K ${num_rows_A} ${partial_output_size} ${block_size} ${split_m} ${single_kernel} 1)

          if [ "$result" == "FAILED" ]; then
            echo "FAILED: ./$exe $M $N $K ${num_rows_A} ${partial_output_size} ${block_size} ${split_m} ${single_kernel} 1"
          fi
        done # block_size
      done # single_kernel
    done # split_m
  done < $fullfil
done # fil


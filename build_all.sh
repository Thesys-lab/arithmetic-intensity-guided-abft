#!/bin/bash

for branch in master abft-thread abft-global; do
  dir=cutlass-$branch
  if [ ! -d $dir/build-h1688 ]; then
    ./cutlass-util/build.sh $dir h1688
  else
    echo "Already built for branch ${branch}"
  fi
done

if [ ! -f cutlass-util/abft_dot/dot_fp16.out ]; then
  curdir=$(pwd)
  cd cutlass-util/abft_dot/
  make
  cd $curdir
fi

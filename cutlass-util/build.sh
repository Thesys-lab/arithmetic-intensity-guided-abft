#!/bin/bash

source $CUTLASS_UTIL_DIR/common.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <branch> <prec>"
  exit 1
fi

branch=$1
prec=$2

branch=$(get_branch $branch)
cd cutlass-$branch

builddir=$(build_dir_for_type $prec)
cmake_args=$(cmake_args_for_type $prec)

if [ ! -d $builddir ]; then mkdir $builddir; fi
cd $builddir
cmake .. $cmake_args
make cutlass_profiler -j$(nproc)

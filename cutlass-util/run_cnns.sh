#!/bin/bash

source $CUTLASS_UTIL_DIR/common.sh

# Round a value up to the nearest multiple of 8
round_up8() {
  python3 -c "print(${1} + (8 - (${1} % 8)) if ${1} % 8 != 0 else ${1})"
}

get_overhead() {
  file_base=$1
  file_other=$2
  file_cmp=$3
  overhead_tmp=$(python3 $CUTLASS_UTIL_DIR/compare_output.py $file_base $file_other $file_cmp)
  python3 -c "print('{:.2f}'.format(${overhead_tmp}))"
}

get_overhead_global() {
  file_base=$1
  file_other=$2
  file_cmp=$3
  overhead_tmp=$(python3 $CUTLASS_UTIL_DIR/compare_output_global.py $file_base $file_other $file_cmp)
  python3 -c "print('{:.2f}'.format(${overhead_tmp}))"
}

MINARG=1
MAXARG=1
if [ $# -lt $MINARG ] || [ $# -gt $MAXARG ]; then
  echo "Usage: $0 outdir"
  exit 1
fi
outdir=$1
mode=h1688

if [ ! -d $outdir ]; then
  mkdir $outdir
fi

# Get full path
cd $outdir
outdir=$(pwd)
cd -

outfile=$outdir/results.csv

baseline_branch=$(get_branch master)
abft_thread_branch=$(get_branch abft-thread) 
abft_global_branch=$(get_branch abft-global)

precision=$(precision_for_type $mode)
precision_name=$(precision_name_for_type $mode)
kernels=$(kernels_for_type $mode)
builddir=$(build_dir_for_type $mode)
iterations=1000
warmup=100
verify=false

baseline_file=$outdir/out-og.csv
abft_thread_file=$outdir/out-abft_thread.csv
abft_global_file=$outdir/out-abft_global.csv
cmp_file=$outdir/out-cmp.csv

curdir=$(pwd)
#gemmdir=$CUTLASS_UTIL_DIR/gemm/noscope
gemmdir=$CUTLASS_UTIL_DIR/gemm/torchvision
header="model,resolution,batch_size,count,M,N,K,AI,time_og,time_dot,slowdown_abft_thread,slowdown_abft_global_with_dot,slowdown_abft_global_nodot"
echo $header >> $outfile
echo $header

for batch_size in 1; do
  for fil in $(ls $gemmdir/*.txt); do
    fil=$(echo $fil | awk -F'/' '{print $NF}')
    model=$(echo $fil | awk -F'_' '{print $1}')
    res=$(echo $fil | awk -F'_' '{print $NF}' | awk -F'.txt' '{print $1}')
    fullfil=$gemmdir/$fil
    idx=0
    fildir=$outdir/$model-$res-batch_size_$batch_size
    if [ ! -d $fildir ]; then
      mkdir $fildir
    fi
    while IFS=' ' read -r M N K count; do
      batch_M=$(($M * $batch_size))

      if [ "$mode" == "h884" ] || [ "$mode" == "igemm" ] || [ "$mode" == "h1688" ]; then
        batch_M=$(round_up8 $batch_M)
        N=$(round_up8 $N)
        K=$(round_up8 $K)
      fi

      rm -f $baseline_file
      rm -f $abft_thread_file
      rm -f $abft_global_file
      rm -f $cmp_file
      cutlass_args="--kernels=${kernels} --profiling-iterations=${iterations} --warmup-iterations=${warmup} --verification-enabled=${verify} --m=${batch_M} --n=${N} --k=${K} ${CUTLASS_PROFILER_ARGS}"
      ai=$(python3 $CUTLASS_UTIL_DIR/calc_ai.py $batch_M $N $K $precision)

      cd cutlass-$baseline_branch/$builddir
      ./tools/profiler/cutlass_profiler $cutlass_args | grep "," > $baseline_file
      cd $curdir
      cd cutlass-$abft_thread_branch/$builddir
      ./tools/profiler/cutlass_profiler $cutlass_args | grep "," > $abft_thread_file
      cd $curdir
      cd cutlass-$abft_global_branch/$builddir
      ./tools/profiler/cutlass_profiler $cutlass_args --abft_next_kernelH=3 --abft_next_kernelW=3 | grep "," > $abft_global_file
      cd $curdir

      # Sweep through various block sizes for performing aggregation dot product
      # for global ABFT.
      cd cutlass-util/abft_dot
      dotfile=$fildir/$idx-dot.csv
      echo "blocks_m,blocks_mn,dot_time" > $dotfile
      min_dot_time=10000000
      for blocks_m in $(python3 $CUTLASS_UTIL_DIR/block_size.py $abft_global_file); do
        for blocks_mn in $(python3 $CUTLASS_UTIL_DIR/block_size.py $abft_global_file --mode blocks_for_block_m --block_m_to_query $blocks_m); do
          dot_time=$(./sweep.sh $precision_name $batch_M $N $K $blocks_m $blocks_m)
          echo "${blocks_m},${blocks_mn},${dot_time}" >> $dotfile
          min_dot_time=$(python3 -c "print(min(${min_dot_time}, ${dot_time}))")
        done # blocks_mn
      done # blocks_m
      dot_time=$min_dot_time
      cd $curdir

      overhead_abft_thread=$(get_overhead $baseline_file $abft_thread_file $cmp_file)
      overhead_abft_global_nodot=$(get_overhead $baseline_file $abft_global_file $cmp_file)

      gemm_time=$(python3 $CUTLASS_UTIL_DIR/get_min.py $baseline_file)
      abft_global_time=$(python3 -c "print((${gemm_time} * ${overhead_abft_global_nodot}) + ${dot_time})")
      overhead_abft_global=$(python3 -c "print('{:.2f}'.format(${abft_global_time} / ${gemm_time}))")
      val="$model,$res,$batch_size,$count,$batch_M,$N,$K,$ai,$gemm_time,$dot_time,$overhead_abft_thread,$overhead_abft_global,$overhead_abft_global_nodot"
      echo $val >> $outfile
      echo $val
      cp $abft_global_file $fildir/$idx-global.csv
      cp $abft_thread_file $fildir/$idx-thread.csv
      cp $baseline_file $fildir/$idx-baseline.csv

      idx=$(($idx + 1))
    done < $fullfil
  done # file
done # batch size

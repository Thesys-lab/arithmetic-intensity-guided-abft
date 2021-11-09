gpu_line=$(nvidia-smi | grep Tesla)
arch=75
if [ "$(echo $gpu_line | grep V100)" != "" ]; then
  arch=70
fi

export CUTLASS_H1688_CMAKE_ARGS="-DCUTLASS_NVCC_ARCHS=${arch} -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8,cutlass_tensorop_h1688gemm_128x256_32x2_nn_align8,cutlass_tensorop_h1688gemm_128x128_32x2_nn_align8,cutlass_tensorop_h1688gemm_64x128_32x2_nn_align8,cutlass_tensorop_h1688gemm_128x64_32x2_nn_align8,cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8,cutlass_tensorop_h1688gemm_256x128_32x2_nt_align8,cutlass_tensorop_h1688gemm_128x256_32x2_nt_align8,cutlass_tensorop_h1688gemm_128x128_32x2_nt_align8,cutlass_tensorop_h1688gemm_64x128_32x2_nt_align8,cutlass_tensorop_h1688gemm_128x64_32x2_nt_align8,cutlass_tensorop_h1688gemm_64x64_32x2_nt_align8,cutlass_tensorop_h1688gemm_256x128_32x2_tn_align8,cutlass_tensorop_h1688gemm_128x256_32x2_tn_align8,cutlass_tensorop_h1688gemm_128x128_32x2_tn_align8,cutlass_tensorop_h1688gemm_64x128_32x2_tn_align8,cutlass_tensorop_h1688gemm_128x64_32x2_tn_align8,cutlass_tensorop_h1688gemm_64x64_32x2_tn_align8,cutlass_tensorop_h1688gemm_256x128_32x2_tt_align8,cutlass_tensorop_h1688gemm_128x256_32x2_tt_align8,cutlass_tensorop_h1688gemm_128x128_32x2_tt_align8,cutlass_tensorop_h1688gemm_64x128_32x2_tt_align8,cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8,cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8"
export CUTLASS_PROFILER_ARGS="--dist=gaussian,mean:0,stddev:0.1,scale:-1 --seed=0"

get_branch() {
  python3 -c "print('-'.join('${1}'.split('cutlass-')[1:]) if 'cutlass-' in '${1}' else '${1}')"
}

build_dir_for_type() {
  echo "build-h1688"
}

precision_for_type() {
  if [ "$1" == "h884" ] || [ "$1" == "h1688" ] || [ "$1" == "hgemm" ]; then
    echo 2
  elif [ "$1" == "igemm" ]; then
    echo 1
  else
    echo 4
  fi
}

precision_name_for_type() {
  if [ "$1" == "h884" ] || [ "$1" == "h1688" ] || [ "$1" == "hgemm" ]; then
    echo "fp16"
  elif [ "$1" == "igemm" ]; then
    echo "int8"
  else
    echo "fp32"
  fi
}

kernels_for_type() {
  if [ "$1" == "h884" ]; then
    echo cutlass_tensorop_h884gemm_*_align8
  elif [ "$1" == "h1688" ]; then
    echo cutlass_tensorop_h1688gemm_*_align8
  elif [ "$1" == "hgemm" ]; then
    echo cutlass_simt_hgemm*
  elif [ "$1" == "igemm" ]; then
    echo cutlass_simt_igemm*
  else
    echo cutlass_simt_sgemm*
  fi
}

gemm_kernels_for_type() {
  # Leave kernels_for_type as is for backward compatability
  echo $(kernels_for_type $1)
}

cmake_args_for_type() {
  echo $CUTLASS_H1688_CMAKE_ARGS
}

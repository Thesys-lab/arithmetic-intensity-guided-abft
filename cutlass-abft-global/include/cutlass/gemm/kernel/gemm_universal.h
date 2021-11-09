/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief 
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the checksums of matrix outputs
template <
  /// Type that output elementes are represented in
  typename ElementC,

  /// Type of output container
  typename FragmentC
>
struct Checksum;

////////////////////////////////
// Specialization for half_t  //
////////////////////////////////

template <typename FragmentC>
struct Checksum <half_t, FragmentC> {
  __device__
  void output_checksum(const FragmentC& output, half_t& output_checksum) {
    Array<half_t, 2> const *ptr_C = reinterpret_cast<Array<half_t, 2> const *>(&output);
    __half2 o_checksum = make_half2(0.f, 0.f);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < output.size()/2; ++i)
      o_checksum = __hadd2(reinterpret_cast<__half2 const &>(ptr_C[i]), o_checksum);
		output_checksum = (half_t)(o_checksum.x + o_checksum.y);
  }

  __device__
  void next_layer_input_checksum(FragmentC& output, const int cols_per_thread, const int rows_per_thread) {

    // All of my attempts to use __half2 arithmetic like in `output_checksum`
    // failed to produce correct values. I think this might be due to
    // some odd issue in translating between __half and cutlass::half_t.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cols_per_thread; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < rows_per_thread; ++j) {
        output[i] = output[i] + output[(cols_per_thread*j) + i];
      }
    }
  }
};

//////////////////////
// Generic version  //
//////////////////////

template <typename ElementC, typename FragmentC>
struct Checksum {
  __device__
  void output_checksum(const FragmentC& output, ElementC& output_checksum) {
    output_checksum = (ElementC)0.f;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < output.size(); i+=2) 
      output_checksum += (output[i] + output[i+1]);
  }

  __device__
  void next_layer_input_checksum(FragmentC& output, const int cols_per_thread, const int rows_per_thread) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cols_per_thread; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < rows_per_thread; ++j) {
        output[i] = output[i] + output[(cols_per_thread*j) + i];
      }
    }
  }
};


template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmUniversal {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode;
    GemmCoord problem_size;
    int batch_count;

    typename EpilogueOutputOp::Params epilogue;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C;
    void * ptr_D;
    void * ptr_D_sum;
    void * ptr_A_checksum;


    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    int lda;
    int ldb;
    int ldc;
    int ldd;

    int abft_batch_size;
    int abft_next_kernelH;
    int abft_next_kernelW;
    int abft_next_strideH;
    int abft_next_strideW;
    int abft_next_paddingH;
    int abft_next_paddingW;
    int abft_next_dilationH;
    int abft_next_dilationW;


    //
    // Methods
    //
    
    Arguments(): 
      mode(GemmUniversalMode::kGemm), 
      batch_count(1), 
      ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr), ptr_D_sum(nullptr), ptr_A_checksum(nullptr),
      abft_batch_size(1), abft_next_kernelH(1), abft_next_kernelW(1),
      abft_next_strideH(1), abft_next_strideW(1), abft_next_paddingH(0), abft_next_paddingW(0),
      abft_next_dilationH(1), abft_next_dilationW(1) { }

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      void * ptr_D_sum,
      void * ptr_A_checksum,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      int lda,
      int ldb,
      int ldc,
      int ldd,
      int abft_batch_size_,
      int abft_next_kernelH_,
      int abft_next_kernelW_,
      int abft_next_strideH_,
      int abft_next_strideW_,
      int abft_next_paddingH_,
      int abft_next_paddingW_,
      int abft_next_dilationH_,
      int abft_next_dilationW_
      ):
      mode(mode), 
      problem_size(problem_size), 
      batch_count(batch_count),
      epilogue(epilogue), 
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D), ptr_D_sum(ptr_D_sum), ptr_A_checksum(ptr_A_checksum),
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C), batch_stride_D(batch_stride_D), 
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd),
      abft_batch_size(abft_batch_size_), abft_next_kernelH(abft_next_kernelH_),
      abft_next_kernelW(abft_next_kernelW_), abft_next_strideH(abft_next_strideH_),
      abft_next_strideW(abft_next_strideW_), abft_next_paddingH(abft_next_paddingH_),
      abft_next_paddingW(abft_next_paddingW_), abft_next_dilationH(abft_next_dilationH_),
      abft_next_dilationW(abft_next_dilationW_)
    {

      CUTLASS_TRACE_HOST("GemmUniversal::Arguments::Arguments() - problem_size: " << problem_size);
      }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.batch_stride_A, args.batch_stride_B);

      return args;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    
    typename EpilogueOutputOp::Params output_op;

    GemmUniversalMode mode;
    int batch_count;
    int gemm_k_size;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;
    void * ptr_D_sum;
    void * ptr_A_checksum;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    int *semaphore;

    int abft_batch_size;
    int abft_next_kernelH;
    int abft_next_kernelW;
    int abft_next_strideH;
    int abft_next_strideW;
    int abft_next_paddingH;
    int abft_next_paddingW;
    int abft_next_dilationH;
    int abft_next_dilationW;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      params_A(0),
      params_B(0),
      params_C(0),
      params_D(0),
      batch_count(0),
      gemm_k_size(0),
      mode(cutlass::gemm::GemmUniversalMode::kGemm),
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr),
      ptr_D_sum(nullptr),
      ptr_A_checksum(nullptr),
      batch_stride_A(0),
      batch_stride_B(0),
      batch_stride_C(0),
      batch_stride_D(0),
      semaphore(nullptr),
      abft_batch_size(1), abft_next_kernelH(1), abft_next_kernelW(1),
      abft_next_strideH(1), abft_next_strideW(1), abft_next_paddingH(0), abft_next_paddingW(0),
      abft_next_dilationH(1), abft_next_dilationW(1) { }

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      int gemm_k_size,
      void *workspace = nullptr
    ):
      problem_size(args.problem_size),
      grid_tiled_shape(grid_tiled_shape),
      params_A(args.lda),
      params_B(args.ldb),
      params_C(args.ldc),
      params_D(args.ldd),
      output_op(args.epilogue),
      mode(args.mode),
      batch_count(args.batch_count),
      gemm_k_size(gemm_k_size),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      ptr_D_sum(args.ptr_D_sum),
      ptr_A_checksum(args.ptr_A_checksum),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      batch_stride_D(args.batch_stride_D),
      semaphore(static_cast<int *>(workspace)),
      abft_batch_size(args.abft_batch_size),
      abft_next_kernelH(args.abft_next_kernelH),
      abft_next_kernelW(args.abft_next_kernelW),
      abft_next_strideH(args.abft_next_strideH),
      abft_next_strideW(args.abft_next_strideW),
      abft_next_paddingH(args.abft_next_paddingH),
      abft_next_paddingW(args.abft_next_paddingW),
      abft_next_dilationH(args.abft_next_dilationH),
      abft_next_dilationW(args.abft_next_dilationW)
    {

      CUTLASS_TRACE_HOST("GemmUniversal::Params::Params() - problem_size: " << problem_size);
    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr) {

      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;
      ptr_D_sum = args.ptr_D_sum;
      ptr_A_checksum = args.ptr_A_checksum;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_D = args.batch_stride_D;

      output_op = args.epilogue;

      semaphore = static_cast<int *>(workspace);

      abft_batch_size = args.abft_batch_size;
      abft_next_kernelH = args.abft_next_kernelH;
      abft_next_kernelW = args.abft_next_kernelW;
      abft_next_strideH = args.abft_next_strideH;
      abft_next_strideW = args.abft_next_strideW;
      abft_next_paddingH = args.abft_next_paddingH;
      abft_next_paddingW = args.abft_next_paddingW;
      abft_next_dilationH = args.abft_next_dilationH;
      abft_next_dilationW = args.abft_next_dilationW;

      CUTLASS_TRACE_HOST("GemmUniversal::Params::update()");
    }
  };


  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmUniversal() { } 

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmUniversal::can_implement()");

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A); 
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    ElementC *ptr_D_sum = static_cast<ElementC *>(params.ptr_D_sum);
    ElementA *ptr_A_checksum = static_cast<ElementA *>(params.ptr_A_checksum);

    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm || 
      params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size; 
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[threadblock_tile_offset.k()];
    }

    __syncthreads();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{
      offset_k,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };


    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    #define SHFL_MASK 0xffffffff
    int warp_idx = __shfl_sync(SHFL_MASK, threadIdx.x / 32, 0);

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //
    
    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations, 
      accumulators, 
      iterator_A, 
      iterator_B, 
      accumulators);

    // Compute thread-local output checksum
    typename Mma::FragmentC::Element output_sum(0.f);
    Checksum<typename Mma::FragmentC::Element, typename Mma::FragmentC> checksum_generator;
    checksum_generator.output_checksum(accumulators, output_sum);

    // Compute threadblock-level output checksum using existing shared storage.
    // TODO: Make this work in the (unlikely) case that the epilogue shared-mem
    // space is less than the number of threads in the block.
    auto output_sum_shared = shared_storage.epilogue.storage.data();
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    output_sum_shared[tid] = output_sum;
    __syncthreads();

    // The algorithm used here follows from:
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf (slide 35)
    if (blockSize >= 512) { if (tid < 256) { output_sum_shared[tid] += output_sum_shared[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { output_sum_shared[tid] += output_sum_shared[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { output_sum_shared[tid] += output_sum_shared[tid +  64]; } __syncthreads(); }

    if (blockSize >= 64)  {
      // From slide 31 and 32 of the link above, all threads tid < 32 are
      // in same warp and so will run in lock step. Thus, we do not need
      // to call __syncthreads for these.
      //
      // In my experience, the calls to __syncthreads are actually necessary
      // to get the correct final result.
      //
      // I tried using __shfl intrinsics, but got worse performance.
      if (tid < 32)  { output_sum_shared[tid] += output_sum_shared[tid +  32]; } __syncthreads();
      if (tid < 16)  { output_sum_shared[tid] += output_sum_shared[tid +  16]; } __syncthreads();
      if (tid < 8)  { output_sum_shared[tid] += output_sum_shared[tid +  8]; } __syncthreads();
      if (tid < 4)  { output_sum_shared[tid] += output_sum_shared[tid +  4]; } __syncthreads();
      if (tid < 2)  { output_sum_shared[tid] += output_sum_shared[tid +  2]; } __syncthreads();
      if (tid < 1)  { output_sum_shared[tid] += output_sum_shared[tid +  1]; }
    }
    __syncthreads();

    // NOTE: Don't change this block_idx. It was set (and used) by CUTLASS
    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();
    if (threadIdx.x == 0) {
      //printf("Writing %f to %d\n", output_sum_shared[0], blockNum);
      ptr_D_sum[block_idx] = output_sum_shared[0];
    }
    __syncthreads();

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C); 
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    //
    // Fetch pointers based on mode.
    //
    
    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == GemmUniversalMode::kGemm) {

      // If performing a reduction via split-K, fetch the initial synchronization
      if (params.grid_tiled_shape.k() > 1) {
        
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      }
    }
    else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C += threadblock_tile_offset.k() * params.batch_stride_C;
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }


    // Execute the epilogue operator to update the destination tensor.
    epilogue(
      output_op, 
      iterator_D, 
      accumulators, 
      iterator_C);

    // How much shared memory do we need?
    //
    // We know that we need to produce N * Kh * Kw checksum values for the
    // "global" checksum. We will assume that Kh = Kw = 3 for now.
    //
    // Each block produces an Mb * Nb potion of the output. The overall M
    // dimension of the matrix corresponds to `batch_size * height * width`,
    // and each column corresponds to an output channel from this convolution.
    // Thus, we can think of each row of the output matrix as corresponding
    // to an individual pixel (or general 2d activation location). Each "pixel"
    // will appear in Kh * Kw rows of the next layer's input (ignoring boundary
    // conditions).
    //
    // Consider each row in the next layer's input as being (Kh * Kw = 9)
    // sub-rows of length N. Within each sub-row, certain column values
    // will be contributed by the same threadblock. For example, the first
    // Nb columns of each sub-row will be contributed by blocks with
    // block N index of 0, the second by those with N index of 1, and so on.
    //
    // Thus, each block needs a resultant checksum of size Nb * Kh * Kw. This
    // is the minimum amount that must be kept in shared memory.
    //
    // Each block has Mb rows that need to be placed in the Kh * Kw copies of
    // Nb. Since each pixel entry should appear in each of the Kh * Kw spots exactly
    // once, we can simply perform a single checksum of size Nb, and save this
    // to global memory. Then, the dot product kernel can load in Kh * Kw copies
    // of these as need be.
    //
    // Note that this only works when there is enough padding in the next
    // layer's convolution to ensure that every pixel is used exactly Kh * Kw
    // times, and when the stride is equal to 1. Dealing with padding etc. will
    // make things more complicated (and likely slower). For now, I will assume the
    // best-case for this approach.

    // Generate partial checksums for the next layer's A.
    // This is not technically correct because `accumulators` is read-only 
    // in the call to the epilogue function above.
    // The epilogue also (internally) rearranges which output elements are
    // stored by each thread. As a result, what we are accumulating below
    // is really just the pre-epilogue accumulators. While not technically
    // correct, it will provide a best-case performance estimate for global ABFT
    // (which is likely pretty tight).
    const int cols_per_thread = Mma::Operator::FragmentB::kElements;
    const int rows_per_thread = Mma::Operator::FragmentA::kElements;

    checksum_generator.next_layer_input_checksum(accumulators, cols_per_thread, rows_per_thread);

    // TODO: Change this for cases where we can't first do a thread-local
    // partial checksum.
    const int rows_to_save_per_thread = 1;

    const int warp_rows = ThreadblockShape::kM / WarpShape::kM;

    // Below, we set up the addresses to be saved to within shared memory. This
    // calculation is not of the rows/columns that each thread contains results
    // for, but, rather, which indices it should save to for performing
    // a block-level reduction.
    //
    // Our aim here is to construct a `(Mb / rows_to_save_per_thread) x Nb`
    // matrix in shared memory containing thread-local partial summations.
    // After forming this matrix, individual threads be responsible for forming
    // the resultant checksum for each of the Nb columns by reading each
    // element of the column.

    // There are `warp_rows` warps per warp column, so the warp column id of
    // this warp is warp_idx / warp_rows.
    const int warp_col_id = warp_idx / warp_rows;

    // There are `warp_rows` warps per column, so our warp row is
    // `warp_idx % warp_rows`.
    const int warp_row_id = warp_idx % warp_rows;

    // Calculate the starting column index for this thread to write to within
    // shared memory. This boils down to calculating the thread's index within
    // the overall threadblock. We decompose this into (1) calculating the
    // thread's starting column index within its warp, and (2) calculating
    // the warp's starting column index within the block.

    // We assume the following layout for threads within a warp, which has been
    // determined by profiling the indices accessed by threads in CUTLASS. In
    // each case, a thread contains `cols_per_thread` columns and `rows_per_thread`
    // rows.
    //
    // WarpShape::<M=32,N=64>:
    // 00 02 04 06 08 10 12 14
    // 01 03 05 07 09 11 13 15
    // 16 18 20 22 24 26 28 30
    // 17 19 21 23 25 27 29 31
    //
    // WarpShape::<M=64,N=32>:
    // 00 02 04 06
    // 01 03 05 07
    // 08 10 12 14
    // 09 11 13 15
    // 16 18 20 22
    // 17 19 21 23
    // 24 26 28 30
    // 25 27 29 32

    // Calculate the starting column index for the thread in its warp. As shown
    // above, pairs of consecutive threads are located within the same column.
    // In a given row, a warp has `WarpShape::kN / cols_per_thread` threads.
    // The starting column index for this thread is thus determined by (1)
    // which thread pair the thread belongs to (thread_idx / 2), and (2)
    // which index within the warp that pair belongs to
    // ((thread_idx / 2) % threads_per_warp_row).
    const int threads_per_warp_row = (WarpShape::kN / cols_per_thread);
    const int thread_col_in_warp = ((thread_idx / 2) % threads_per_warp_row) * cols_per_thread;

    // The overall starting column index within this block is then the
    // addition of this thread's warp's column offset (warp_col_id * WarpShape::kN)
    // and this thread's starting column offset within the warp.
    const int warp_col_in_block = warp_col_id * WarpShape::kN;
    const int my_col = warp_col_in_block + thread_col_in_warp;

    // Calculate the starting row index for the thread. We, again, decompose
    // this into finding the starting row index for this thread's warp and
    // that for the thread within the warp.
    
    // Each warp has WarpShape::KM / rows_per_thread threads in each column.
    // Each of those threads will save `rows_to_save_per_thread` values, each
    // of which needs its own entry in the column (and thus is on a separate
    // row). Thus, each warp will save to
    // `WarpShape::kM / rows_per_thread * rows_to_save_per_thread` rows.
    // The starting row index for this warp is then just this value multiplied
    // by the warp's row index.
    const int threads_per_col_in_warp = WarpShape::kM / rows_per_thread;
    const int rows_to_save_per_warp = threads_per_col_in_warp * rows_to_save_per_thread;
    const int warp_row_in_block = warp_row_id * rows_to_save_per_warp;

    // Calculate this thread's starting row index within its warp. In performing
    // this calculation, we use the thread's index within its warp (`shifted_tid`
    // below). We first calculate this thread's offset within the warp ignoring
    // the number of rows each thread will save, and then multiply this resultant
    // value by the number of rows saved by each thread (rows_to_save_per_thread).
    //
    // This thread's row within its warp is determined by (1) which pair it
    // belongs to, (2) the starting row index for that pair, and (3) which
    // thread within the pair the thread is.
    //
    // (1) is determined by (shifted_tid / 2).
    //
    // (2) is determined by the number of pairs per row. There are 
    // `threads_per_warp_row` threads in each row (and thus pairs per row). So
    // the pair's starting row offset is determined by (pair_id / threads_per_warp_row) * 2
    //
    // (3) is determined by (shifeted_tid % 2).
    //
    // We combine these values and multiply by `rows_to_save_per_thread`.
    const int shifted_tid = thread_idx - (32 * warp_idx);
    const int threads_per_pair = 2;
    const int pair_id = shifted_tid / threads_per_pair;
    const int id_in_pair = shifted_tid % threads_per_pair;
    const int thread_row_in_warp = (((pair_id / threads_per_warp_row) * threads_per_pair) + id_in_pair) * rows_to_save_per_thread;
    const int my_row = (warp_row_in_block + thread_row_in_warp) * ThreadblockShape::kN;

    auto shared_scratchpad = shared_storage.epilogue.storage.data();

    // We now have `cols_per_thread` partial accumulations per thread. We need
    // to write these to relevant locations of shared memory so that we can
    // perform our reduction.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rows_to_save_per_thread; ++i) {
      int row_id = my_row + (i * ThreadblockShape::kN);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < cols_per_thread; ++j) {
        int col_id = my_col + j;
        int accum_id = (i * cols_per_thread) + j;
        shared_scratchpad[row_id + col_id] = accumulators[accum_id];
      }
    }
    __syncthreads();

    const int global_row = threadblock_tile_offset.m();
    const int global_col = threadblock_tile_offset.n();

    const int repetitions = params.abft_next_kernelH * params.abft_next_kernelW;

    // Shared memory now contains a (Mb/rows_per_thread) x Nb
    // matrix. We need to reduce this column-wise. To do so, we assign one thread
    // per column and iteratively load and form partial checksums.
    for (int block_it = 0; block_it < ThreadblockShape::kN; block_it += blockDim.x) {
      const int col_id = block_it + threadIdx.x;
      if (col_id < ThreadblockShape::kN) {
        typename Mma::FragmentC::Element achecksum_col_sum(0.f);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < (ThreadblockShape::kM / rows_per_thread * rows_to_save_per_thread); ++i) {
          achecksum_col_sum += shared_scratchpad[(i * ThreadblockShape::kN) + col_id];
        }

        // Write the result to global memory. For now, we do this in
        // Nb * Kh * Kw locations. In reality, this will need to happen
        // for differnt partial summations, as each of these entries will
        // have some components left out.
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < repetitions; ++j) {
          const int addr = (global_row * params.problem_size.n() * repetitions) + (j * params.problem_size.n()) + (global_col * ThreadblockShape::kN) + col_id;
          ptr_A_checksum[addr] = achecksum_col_sum;
        }
      }
    }

    //
    // Release the semaphore
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) { 

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }
      
      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

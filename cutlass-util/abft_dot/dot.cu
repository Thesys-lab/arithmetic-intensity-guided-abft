// ./dot.out 1024 1024 1024 2 64 512

#include <iostream>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#ifdef TYPE_FP16
  #define AB_TYPE half
  #define OUT_TYPE half
#endif

#ifdef TYPE_FP32
  #define AB_TYPE float
  #define OUT_TYPE float
#endif

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
      exit(1);
   }
}

__device__
void block_row_start_end(int total_rows, int split, int* start, int* end) {
  int base_rows = total_rows / split;

  // Account for any remaining rows that need to be covered. Note that
  // block_row_id will be less than split
  int my_rows = base_rows;
  int remainder = total_rows % split;
  if (blockIdx.y < remainder)
    my_rows += 1;

  int num_added_before_me = min(blockIdx.y, remainder);
  *start = (base_rows * blockIdx.y) + num_added_before_me;
  *end = *start + my_rows;
  if (*end > total_rows)
    *end = total_rows;
}

template <unsigned int rows>
__device__
AB_TYPE load_and_mult_unrolled(
    const AB_TYPE* A_checksum,
    const AB_TYPE* B_checksum,
    int K,
    int start,
    int end) {

  AB_TYPE total_val = (AB_TYPE)0.f;
  int total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  // We wrap this around in a loop to handle cases where we launch less
  // threads than there are columns. When this is the case, each thread
  // will perform at least one reduction and dot product. If the thread
  // performs more than one, the results will be aggregated together.
  for (int col_start = 0; col_start < K; col_start += total_threads) {
    int col_id = (blockDim.x * blockIdx.x) + threadIdx.x + col_start;
    AB_TYPE val = (AB_TYPE)0.f;
    if (col_id < K) {
      AB_TYPE B_val = B_checksum[col_id];
      #pragma unroll
      for (int i = 0; i < rows; ++i) {
        val += A_checksum[(K * (start + i)) + col_id];
      }

      // Perform any remaining loads
      for (int i = start + rows; i < end; ++i) {
        val += A_checksum[(K * i) + col_id];
      }

      val *= B_val;
      total_val += val;
    }
  }
  return total_val;
}

__device__
AB_TYPE load_and_mult(const AB_TYPE* A_checksum,
                    const AB_TYPE* B_checksum,
                    int K,
                    int A_rows,
                    int split_m) {
  //int col_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  int start = 0, end = 0;
  block_row_start_end(A_rows, split_m, &start, &end);

  int rows = end - start;
  if (rows >= 128)
    return load_and_mult_unrolled<128>(A_checksum, B_checksum, K, start, end);
  else if (rows >= 64)
    return load_and_mult_unrolled<64>(A_checksum, B_checksum, K, start, end);
  else if (rows >= 32)
    return load_and_mult_unrolled<32>(A_checksum, B_checksum, K, start, end);
  else if (rows >= 16)
    return load_and_mult_unrolled<16>(A_checksum, B_checksum, K, start, end);
  else if (rows >= 8)
    return load_and_mult_unrolled<8>(A_checksum, B_checksum, K, start, end);
  else if (rows >= 4)
    return load_and_mult_unrolled<4>(A_checksum, B_checksum, K, start, end);
  else if (rows >= 2)
    return load_and_mult_unrolled<2>(A_checksum, B_checksum, K, start, end);
  else
    return load_and_mult_unrolled<1>(A_checksum, B_checksum, K, start, end);
}

template <unsigned int BlockSize>
__global__
void reduce_and_dot(const AB_TYPE* A_checksum,
                    const AB_TYPE* B_checksum,
                    int K,
                    int A_rows,
                    int split_m,
                    OUT_TYPE* output) {
  int bid = blockIdx.y * gridDim.x + blockIdx.x;
  AB_TYPE val = load_and_mult(A_checksum, B_checksum, K, A_rows, split_m);

  __syncthreads();
  typedef cub::BlockReduce<AB_TYPE, BlockSize> BlockReduce; 
  __shared__ typename BlockReduce::TempStorage temp_storage;
  AB_TYPE aggregate = BlockReduce(temp_storage).Sum(val);
  if (threadIdx.x == 0) {
    output[bid] = (OUT_TYPE)aggregate;
  }
}

__global__
void reduce_and_dot_warp(const AB_TYPE* A_checksum,
                    const AB_TYPE* B_checksum,
                    int K,
                    int A_rows,
                    int split_m,
                    OUT_TYPE* output) {
  int bid = blockIdx.y * gridDim.x + blockIdx.x;
  AB_TYPE val = load_and_mult(A_checksum, B_checksum, K, A_rows, split_m);

  __syncthreads();
  typedef cub::WarpReduce<AB_TYPE> WarpReduce; 
  __shared__ typename WarpReduce::TempStorage temp_storage;
  AB_TYPE aggregate = WarpReduce(temp_storage).Sum(val);
  if (threadIdx.x == 0) {
    output[bid] = (OUT_TYPE)aggregate;
  }
}

template <unsigned int BlockSize>
__global__
void global_reduce(
    const OUT_TYPE* dot_partial,
    const OUT_TYPE* out_partial,
    int dot_size,
    int out_size,
    int* error) {
  typedef cub::BlockReduce<OUT_TYPE, BlockSize> BlockReduce; 
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int bid = blockIdx.x;
  int tid = (blockDim.x * bid) + threadIdx.x;
 
  OUT_TYPE val_dot = (OUT_TYPE)0.f;
  for (int i = 0; i < dot_size; i += blockDim.x) {
    int id = tid + i;
    if (id < dot_size)
      val_dot += dot_partial[id];
  }

  __syncthreads();

  val_dot = BlockReduce(temp_storage).Sum(val_dot);

  OUT_TYPE val_out = (OUT_TYPE)0.f;
  for (int i = 0; i < out_size; i += blockDim.x) {
    int id = tid + i;
    if (id < out_size)
      val_out += out_partial[id];
  }

  __syncthreads();

  val_out = BlockReduce(temp_storage).Sum(val_out);

  if (threadIdx.x == 0) {
    //printf("blockDim.x=%d, Dot val: %f Out val: %f\n", blockDim.x, (float)val_dot, (float)val_out);
    *error = int(val_dot != val_out);
  }
}

__global__
void global_reduce_warp(
    const OUT_TYPE* dot_partial,
    const OUT_TYPE* out_partial,
    int dot_size,
    int out_size,
    int* error) {
  typedef cub::WarpReduce<OUT_TYPE> WarpReduce; 
  __shared__ typename WarpReduce::TempStorage temp_storage;

  int bid = blockIdx.x;
  int tid = (blockDim.x * bid) + threadIdx.x;
  
  OUT_TYPE val_dot = (OUT_TYPE)0.f;
  for (int i = 0; i < dot_size; i += blockDim.x) {
    int id = tid + i;
    if (id < dot_size)
      val_dot += dot_partial[id];
  }

  __syncthreads();

  val_dot = WarpReduce(temp_storage).Sum(val_dot);

  OUT_TYPE val_out = (OUT_TYPE)0.f;
  for (int i = 0; i < out_size; i += blockDim.x) {
    int id = tid + i;
    if (id < out_size)
      val_out += out_partial[id];
  }

  __syncthreads();

  val_out = WarpReduce(temp_storage).Sum(val_out);

  if (threadIdx.x == 0) {
    //printf("Dot val: %f Out val: %f\n", (float)val_dot, (float)val_out);
    *error = int(val_dot != val_out);
  }
}

template <unsigned int BlockSize>
__global__
void global_reduce_dot_and_check(
    const AB_TYPE* A_checksum,
    const AB_TYPE* B_checksum,
    int K,
    int A_rows,
    int split_m,
    const OUT_TYPE* out_partial,
    int out_size,
    int* error) {
  
  OUT_TYPE val_dot = (OUT_TYPE)load_and_mult(A_checksum, B_checksum, K, A_rows, split_m);
  typedef cub::BlockReduce<OUT_TYPE, BlockSize> BlockReduce; 
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __syncthreads();

  val_dot = BlockReduce(temp_storage).Sum(val_dot);

  // Aggregate output
  int bid = blockIdx.x;
  int tid = (blockDim.x * bid) + threadIdx.x;
 
  // Aggregate output
  OUT_TYPE val_out = (OUT_TYPE)0.f;
  for (int i = 0; i < out_size; i += blockDim.x) {
    int id = tid + i;
    if (id < out_size)
      val_out += out_partial[id];
  }

  __syncthreads();

  val_out = BlockReduce(temp_storage).Sum(val_out);

  if (threadIdx.x == 0) {
    //printf("blockDim.x=%d, Dot val: %f Out val: %f\n", blockDim.x, (float)val_dot, (float)val_out);
    *error = int(val_dot != val_out);
  }
}

__global__
void global_reduce_dot_and_check_warp(
    const AB_TYPE* A_checksum,
    const AB_TYPE* B_checksum,
    int K,
    int A_rows,
    int split_m,
    const OUT_TYPE* out_partial,
    int out_size,
    int* error) {
  
  OUT_TYPE val_dot = (OUT_TYPE)load_and_mult(A_checksum, B_checksum, K, A_rows, split_m);
  typedef cub::WarpReduce<OUT_TYPE> WarpReduce; 
  __shared__ typename WarpReduce::TempStorage temp_storage;

  __syncthreads();

  val_dot = WarpReduce(temp_storage).Sum(val_dot);

  // Aggregate output
  int bid = blockIdx.x;
  int tid = (blockDim.x * bid) + threadIdx.x;
 
  // Aggregate output
  OUT_TYPE val_out = (OUT_TYPE)0.f;
  for (int i = 0; i < out_size; i += blockDim.x) {
    int id = tid + i;
    if (id < out_size)
      val_out += out_partial[id];
  }

  __syncthreads();

  val_out = WarpReduce(temp_storage).Sum(val_out);

  if (threadIdx.x == 0) {
    //printf("blockDim.x=%d, Dot val: %f Out val: %f\n", blockDim.x, (float)val_dot, (float)val_out);
    *error = int(val_dot != val_out);
  }
}


cudaError_t run(
    int blocks,
    int blocks_x,
    int blocks_y,
    int block_size,
    const AB_TYPE* a_partial_checksum,
    const AB_TYPE* b_checksum,
    OUT_TYPE* dot_out,
    const OUT_TYPE* out_partial_checksum,
    int K,
    int num_rows_A,
    int split_m,
    int out_MN_reduced,
    int* error,
    bool single_kernel) {

  if (!single_kernel) {
    dim3 grid(blocks_x, blocks_y, 1);
    switch (block_size) {
      case 512:
        reduce_and_dot<512><<<grid, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, dot_out);
        break;
      case 256:
        reduce_and_dot<256><<<grid, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, dot_out);
        break;
      case 128:
        reduce_and_dot<128><<<grid, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, dot_out);
        break;
      case 64:
        reduce_and_dot<64><<<grid, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, dot_out);
        break;
      case 32:
        reduce_and_dot_warp<<<grid, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, dot_out);
        break;
      default:
        printf("Invalid block_size %d\n", block_size);
        exit(1);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
      return err;

    int threads_for_global = max(out_MN_reduced, blocks);
    if (threads_for_global > 256) {
      global_reduce<512><<<1,512>>>(dot_out, out_partial_checksum, blocks, out_MN_reduced, error);
    }
    else if (threads_for_global > 128) {
      global_reduce<256><<<1,256>>>(dot_out, out_partial_checksum, blocks, out_MN_reduced, error);
    }
    else if (threads_for_global > 64) {
      global_reduce<128><<<1,128>>>(dot_out, out_partial_checksum, blocks, out_MN_reduced, error);
    }
    else if (threads_for_global > 32) {
      global_reduce<64><<<1,64>>>(dot_out, out_partial_checksum, blocks, out_MN_reduced, error);
    }
    else {
      global_reduce_warp<<<1,32>>>(dot_out, out_partial_checksum, blocks, out_MN_reduced, error);
    }
  } else { // single_kernel
    switch (block_size) {
      case 512:
        global_reduce_dot_and_check<512><<<1, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, out_partial_checksum, out_MN_reduced, error);
        break;
      case 256:
        global_reduce_dot_and_check<256><<<1, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, out_partial_checksum, out_MN_reduced, error);
        break;
      case 128:
        global_reduce_dot_and_check<128><<<1, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, out_partial_checksum, out_MN_reduced, error);
        break;
      case 64:
        global_reduce_dot_and_check<64><<<1, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, out_partial_checksum, out_MN_reduced, error);
        break;
      case 32:
        global_reduce_dot_and_check_warp<<<1, block_size>>>(a_partial_checksum, b_checksum, K, num_rows_A, split_m, out_partial_checksum, out_MN_reduced, error);
        break;
      default:
        printf("Invalid block_size %d\n", block_size);
        exit(1);
    }
  }

  return cudaGetLastError();
}


template <typename T>
__global__ void initialize_to_val (T *arr, int len, T val) {
  int tid = (blockIdx.x * gridDim.x * blockDim.x) + threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < len; ++i)
      arr[i] = val;
  }
}


int main(int argc, char** argv) {
  int min_arg = 10;
  if (argc < min_arg || argc > min_arg + 2) {
    printf("Usage: %s <M> <N> <K> <num_rows_A> <num_partial_output_reduction> <block_size> <split_m> <single_kernel> <is_test> [<iterations> <warmup-iterations>]\n", argv[0]);
    exit(1);
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int num_rows_A = atoi(argv[4]);
  int num_partial_output_reduction = atoi(argv[5]);
  int block_size = atoi(argv[6]);
  int split_m = atoi(argv[7]);
  bool single_kernel = (atoi(argv[8]) == 1);
  bool is_test = (atoi(argv[9]) == 1);

  if (split_m < 1 || split_m > num_rows_A) {
    printf("<split_m> must be between 1 and num_rows_A=%d. Got split_m=%d\n",
        num_rows_A, split_m);
    exit(1);
  }

  int blocks_x = (int)ceil((float)K / (float)block_size);
  int blocks_y = split_m;
  int blocks = blocks_x * blocks_y;

  int iterations = 1000;
  int warmup_iterations = 100;
  if (argc > min_arg)
    iterations = atoi(argv[min_arg]);
  if (argc > min_arg+1)
    warmup_iterations = atoi(argv[min_arg+1]); 

  AB_TYPE* a_partial_checksum;
  AB_TYPE* b_checksum;
  OUT_TYPE* out_partial_checksum;
  OUT_TYPE* dot_out;
  int* error;

  int partial_a_size = num_rows_A * K;
  cudaErrCheck(cudaMalloc((void**)&a_partial_checksum, partial_a_size * sizeof(AB_TYPE)));
  cudaErrCheck(cudaMalloc((void**)&b_checksum,  K * sizeof(AB_TYPE)));
  cudaErrCheck(cudaMalloc((void**)&out_partial_checksum, num_partial_output_reduction * sizeof(OUT_TYPE)));
  cudaErrCheck(cudaMalloc((void**)&dot_out, blocks * sizeof(OUT_TYPE)));
  cudaErrCheck(cudaMalloc((void**)&error, sizeof(int)));


  initialize_to_val<AB_TYPE><<<1,1>>>(a_partial_checksum, partial_a_size, (AB_TYPE)1.0f);
  initialize_to_val<AB_TYPE><<<1,1>>>(b_checksum, K, (AB_TYPE)1.0f);
  initialize_to_val<OUT_TYPE><<<1,1>>>(dot_out, blocks, (OUT_TYPE)0.0f);

  // Since we initialize A to all ones, the result of its checksum will be a 
  // row vector with each entry being of value `num_rows_A`.
  // Since we initialize b_checksum to all ones, the result of the checksum
  // dot product will be `num_rows_A * K`.
  // Thus, for the checksum to work out that `val * num_partial_output_reduction = num_rows_A * K`,
  // we need to have `val = (num_rows_A * K) / num_partial_output_reduction`.
  float out_init_val = (float)(num_rows_A * K) / (float)num_partial_output_reduction;
  initialize_to_val<OUT_TYPE><<<1,1>>>(out_partial_checksum, num_partial_output_reduction, (OUT_TYPE)out_init_val);

  if (is_test) {
    cudaErrCheck(run(blocks, blocks_x, blocks_y,
                 block_size, a_partial_checksum, b_checksum, dot_out,
                 out_partial_checksum, K, num_rows_A, split_m,
                 num_partial_output_reduction, error, single_kernel));
    cudaDeviceSynchronize();
    int host_error;
    cudaErrCheck(cudaMemcpy(&host_error, error, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_error == 0)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    return host_error;
  } else {
    for (int i = 0; i < warmup_iterations; ++i)
      cudaErrCheck(run(blocks, blocks_x, blocks_y,
                   block_size, a_partial_checksum, b_checksum, dot_out,
                   out_partial_checksum, K, num_rows_A, split_m,
                   num_partial_output_reduction, error, single_kernel));
    cudaDeviceSynchronize();

    cudaEvent_t start_time;
    cudaEvent_t stop_time;
    cudaErrCheck(cudaEventCreate(&start_time));
    cudaErrCheck(cudaEventCreate(&stop_time)); 

    cudaErrCheck(cudaEventRecord(start_time));
    for (int i = 0; i < iterations; ++i)
      cudaErrCheck(run(blocks, blocks_x, blocks_y,
                   block_size, a_partial_checksum, b_checksum, dot_out,
                   out_partial_checksum, K, num_rows_A, split_m,
                   num_partial_output_reduction, error, single_kernel));

    cudaErrCheck(cudaEventRecord(stop_time));
    int host_error;
    cudaErrCheck(cudaMemcpy(&host_error, error, sizeof(int), cudaMemcpyDeviceToHost));
    float total_time;
    cudaErrCheck(cudaEventSynchronize(stop_time));

    // Calculate time in milliseconds
    cudaErrCheck(cudaEventElapsedTime(&total_time, start_time, stop_time));
    float avg_time = total_time / (float)iterations;
    printf("Time=%fms\n", avg_time);

    int bytes = ((partial_a_size  + K) * sizeof(AB_TYPE)) + ((blocks + num_partial_output_reduction) * sizeof(OUT_TYPE));
    double gbytes = bytes / 1024.f / 1024.f / 1024.f;
    double sec = (avg_time / 1000.0f);
    printf("GiB/s=%f\n", gbytes / sec);
  }

  //printf("Error value: %d\n", host_error);

  return 0;
}

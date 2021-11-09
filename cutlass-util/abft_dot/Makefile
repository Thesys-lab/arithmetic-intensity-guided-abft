all: dot_fp32 dot_fp16

dot_fp32: dot.cu
	nvcc -DTYPE_FP32 -arch=sm_75 dot.cu -o dot_fp32.out

dot_fp16: dot.cu
	nvcc -DTYPE_FP16 -arch=sm_75 dot.cu -o dot_fp16.out

clean:
	rm dot_fp32.out dot_fp16.out

# Arithmetic-intensity-guided ABFT
This repository contains the code and scripts used in evaluating the SC 2021
paper titled "Arithmetic-Intensity-Guided Fault Tolerance for Neural
Network Inference on GPUs".

## Contents of the repository
* [cutlass-util](cutlass-util): directory containing scripts for building
  and running experiments.
* [cutlass-master](cutlass-master): directory containing the CUTLASS branch
  that we compare against in the absence of fault tolerance. This directory
  is an unmodified snapshot of the upstream CUTLASS repository.
* [cutlass-abft-thread](cutlass-abft-thread): directory containing the
  implementation of thread-level ABFT that we develop.
* [cutlass-abft-global](cutlass-abft-global): directory containing the
  implementation of global ABFT that we use as part of AI-guided ABFT.

## Evaluation environment
* AWS g4dn.xlarge instance: T4 GPU, 4-core Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz
* NVIDIA driver version 450.51.06
* Ubuntu 18.04 running Linux kernel 5.30, NVIDIA HPC SDK AMI VM
* NVCC 11.0
* CUDA 11.0
* NVIDIA-docker using nvidia/cuda:11.0-devel base image

## Setting up the Docker container
From within this top-level directory, run:
```bash
./run_docker.sh
```

This should pull and build the required Docker container (if it has not been built
already), as well as start the required Docker container. You can test whether your
setup has correct access to the GPU by running `nvidia-smi`.

## Building CUTLASS branches
Once you have started the required Docker container, you must build the implementations
used in evaluation. To do so, perform the following commands:
```bash
cd /home/cutlass-util
source setup.sh
cd /home
./build_all.sh
```

## Running experiments
To profile the DLRMs used in our evaluation, run the following from within the Docker container:
```bash
cd /home
./cutlass-util/run_dlrms.sh /path/to/save
```
This will begin profiling each of the matrix multiplications in each DLRM considered, print the
slowdowns resulting from each ABFT version, and save results to the directory specified in
`/path/to/save`.

You should see results being printed like the following:
```
model,batch_size,count,M,N,K,AI,time_og,time_dot,slowdown_abft_thread,slowdown_abft_global_with_dot,slowdown_abft_global_nodot
mlp-bot.txt,1,1,8,512,16,5.28,0.00558326,0.004144,1.06,2.04,1.30
mlp-bot.txt,1,1,8,256,512,7.64,0.0174184,0.005034,1.05,1.39,1.10
...
```
Each line indicates the execution-time overhead for thread-level ABFT and global ABFT for
a given layer of `model`. For global ABFT, we print both the execution-time overhead when
performing the separate comparison dot product kernel to aggregate partial global summations
(`slowdown_abft_global_with_dot`) and that without this dot product kernel (`slowdown_abft_global_nodot`). 
Note that the results in the paper use `slowdown_abft_global_nodot` for global ABFT for all but the final layer,
as these extra dot product kernels can be overlapped with the subsequent layer's execution for all
but the final layer.

To profile the CNNs used in our evaluation, replace `run_dlrms.sh` above with `run_cnns.sh`.

To change which matrix multiplications are profiled (i.e., using different workloads from Sec. 6.2
of the paper), change the directory pointed to by `gemmdir` in `run_cnns.sh` and `run_dlrms.sh`. Relevant
directories are located in [cutlass-util/gemm](cutlass-util/gemm), with `torchvision` and `noscope` being
used in `run_cnns.sh`, and `dlrm` and `mm` in `run_dlrms.sh`.

## Support
We graciously acknowledge support from a National Science Foundation 
(NSF) Graduate Research Fellowship (DGE-1745016 and DGE-1252522), Amazon Web Services,
and  he AIDA – Adaptive, Intelligent and Distributed Assurance Platform – project (reference
POCI-01-0247-FEDER-045907) co-financed by the ERDF - European Regional Development Fund
through the Operational Program for Competitiveness and Internationalisation - COMPETE 2020.

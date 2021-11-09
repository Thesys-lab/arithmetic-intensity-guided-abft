# GEMM Files
This directory contains files containing the GEMM configurations of layers of
a given neural network.

Each line in each file follows the following format:
```
M N K count
```
where `count` is the number of times that a GEMM of this dimension appears in model.

The `M` dimension listed assumes batch size of 1. You can simply multiply it by
batch size to get the appropriate GEMM for a given batch size.

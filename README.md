# gnNHWC
A collection of highly optimized CUDA kernels for NHWC GroupNorm for PyTorch.

# Motivation
As of January 16, 2024, PyTorch does not have a CUDA implementation for NHWC group normalization (GN). NHWC can often boost a CNN's inference/training speed on GPU by 1.25x because convolutions runs more efficiently using NHWC format. However, since GN doesn't have a NHWC kernel, this speedup is nullified by the memory permutation operators to convert a tensor from NHWC to NCHW, run the GN forward/backward, then convert the output back to NHWC. Furthermore, many widely used models (e.g. Stable Diffusion) use GN rather than BatchNorm, making a NHWC GN the missing link to enabling the 1.25x speedup by running these models in NHWC.

# Kernels
- General purpose kernel (works well for small-large batch sizes with large spatial resolutions)
- Small input kernel (whole input loaded in shared memory allowing 2x fewer GPU device memory accesses -> ~1.75x inference speedup)
- A lot of other ones

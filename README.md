# gnNHWC
A highly optimized CUDA kernel for channels last (NHWC) group normalization in PyTorch.

## Motivation
As of January 16, 2024, PyTorch does not have a CUDA implementation for NHWC group normalization (GN). NHWC can often boost a CNN's inference/training speed on GPU by 1.25x because convolutions runs more efficiently using NHWC format. However, since GN doesn't have a NHWC kernel, this speedup is nullified by the memory permutation operators to convert a tensor from NHWC to NCHW, run the GN forward/backward, then convert the output back to NHWC. Furthermore, many widely used models (e.g. Stable Diffusion) use GN rather than BatchNorm, making a NHWC GN the missing link to enabling the 1.25x speedup by running these models in NHWC. 

## Performance
On a RTX 3060 Max-Q, this implementation runs significantly faster than PyTorch's GroupNorm implementation over a wide range of batch sizes, resolutions, and channel counts. Furthermore, this implementation effectively matches the speed of stable-fast's Triton NHWC GroupNorm implementation when running on Stable Diffusion and beats the Triton version when running it outside of the stable-fast library (i.e. using Triton GN without stable-fast's JIT compilation).

## Todo
- [x] Working forward kernel for all size inputs 
- [x] Working backward kernel for all size inputs 
- [x] Forward/backward kernel surpasses PyTorch GN speed and matches stable-fast Triton GN
- [ ] Python wrapper for fused forward activation (e.g. fused GN + Silu)
- [ ] Backward pass with fused activation gradients
- [ ] TorchDynamo functionality?

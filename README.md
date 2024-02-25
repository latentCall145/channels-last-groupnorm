# gnNHWC
A highly optimized CUDA kernel for channels last (NHWC) group normalization in PyTorch.

## Motivation
As of January 16, 2024, PyTorch does not have a CUDA implementation for NHWC group normalization (GN). NHWC can often boost a CNN's inference/training speed on GPU by 1.25x because convolutions runs more efficiently using NHWC format. However, since GN doesn't have a NHWC kernel, this speedup is nullified by the memory permutation operators to convert a tensor from NHWC to NCHW, run the GN forward/backward, then convert the output back to NHWC. Furthermore, many widely used models (e.g. Stable Diffusion) use GN rather than BatchNorm, making a NHWC GN the missing link to enabling the 1.25x speedup by running these models in NHWC. 

Finally, by fusing the activation with the GN kernel, further speedups and memory savings can be made.

## Use
Just a drop-in replacement for torch.nn.GroupNorm (except the activation keyword, which allows the GN forward/backward to run using a fused activation)

Code:
```python
'''
if your directory structure is like:
gnNHWC
├── custom_gn.py
└── (your files here)
'''
from custom_gn import GN_NHWC # if running from within the gnNHWC folder

'''
if your directory structure is like:
my_project
├── gnNHWC
└── (your files here)
'''
from gnNHWC.custom_gn import GN_NHWC # if running from outside the gnNHWC folder

GN_NHWC(32, 128, activation='identity') # replaces nn.GroupNorm(32, 128)
GN_NHWC(32, 128, activation='relu') # replaces nn.GroupNorm(32, 128) and nn.ReLU()
GN_NHWC(32, 128, activation='silu') # replaces nn.GroupNorm(32, 128) and nn.SiLU()
GN_NHWC(32, 128, activation='gelu') # replaces nn.GroupNorm(32, 128) and nn.GeLU()
GN_NHWC(32, 128, activation='gelu_tanh') # replaces nn.GroupNorm(32, 128) and nn.GeLU(approximate='tanh')
```

## Performance
On a RTX 3060 Max-Q, this implementation runs significantly faster than PyTorch's GroupNorm implementation over a wide range of batch sizes, resolutions, and channel counts. This implementation is about 50% faster than stable-fast's Triton NHWC GroupNorm implementation, with stable-fast's Stable Diffusion demo running 0.5-1% faster end-to-end using my GN kernel compared to its Triton GN kernel (tested using SD1.5 @ 512x512 res, bs=1 on a RTX 3060 Max-Q, SDXL @ 1024x1024 res, bs=1 on an A100 80 GB). In conjunction with NHWC convolution, I get around a 35% end-to-end speedup using my fused GN kernel.

The CUDA kernels also compile about 5x faster than PyTorch's native GN CUDA kernels (from ~50 seconds to ~10 seconds) on my laptop (Intel i7-12700H @ 2.7 GHz CPU) as I removed many torch header files which slowed down compilation time.

## Todo
- [x] Working forward kernel for all size inputs 
- [x] Working backward kernel for all size inputs 
- [x] Forward/backward kernel surpasses PyTorch GN speed and matches stable-fast Triton GN
- [X] Python wrapper for fused forward activation (e.g. fused GN + Silu)
- [x] Backward pass with fused activation gradients
- [ ] TorchDynamo functionality?

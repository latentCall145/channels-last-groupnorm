# gnNHWC
A highly optimized CUDA kernel for channels last (NHWC) group normalization in PyTorch.

## Motivation
As of January 16, 2024, PyTorch does not have a CUDA implementation for NHWC group normalization (GN). NHWC can often boost a CNN's inference/training speed on GPU by 1.25x because Tensor Cores run convolutions in NHWC format. However, since GN doesn't have a NHWC kernel, this speedup is nullified by the memory permutation operators to convert a tensor from NHWC to NCHW, run the GN forward/backward, then convert the output back to NHWC. Furthermore, many widely used models (e.g. Stable Diffusion) use GN rather than BatchNorm, making a NHWC GN the missing link to enabling the 1.25x speedup by running these models in NHWC. 

Finally, by fusing the activation with the GN kernel, further speedups and memory savings can be made.

## Use
Just a drop-in replacement for torch.nn.GroupNorm (except the activation keyword, which allows the GN forward/backward to run using a fused activation)

`pip install git+https://github.com/latentCall145/channels-last-groupnorm.git`

Code:
```python
from gnnhwc import GN_NHWC

GN_NHWC(32, 128, activation='identity') # replaces nn.GroupNorm(32, 128)
GN_NHWC(32, 128, activation='relu') # replaces nn.GroupNorm(32, 128) and nn.ReLU()
GN_NHWC(32, 128, activation='silu') # replaces nn.GroupNorm(32, 128) and nn.SiLU()
GN_NHWC(32, 128, activation='gelu') # replaces nn.GroupNorm(32, 128) and nn.GeLU()
GN_NHWC(32, 128, activation='gelu_tanh') # replaces nn.GroupNorm(32, 128) and nn.GeLU(approximate='tanh')
```

## Performance
On a RTX 3060 Max-Q, this implementation runs significantly faster than PyTorch's GroupNorm implementation over a wide range of batch sizes, resolutions, and channel counts. This implementation is about 50% faster than stable-fast's Triton NHWC GroupNorm implementation, with stable-fast's Stable Diffusion demo running 0.5-1% faster end-to-end using my GN kernel compared to its Triton GN kernel (tested using SD1.5 @ 512x512 res, bs=1 on a RTX 3060 Max-Q, SDXL @ 1024x1024 res, bs=1 on an A100 80 GB). In conjunction with NHWC convolution, I get around a 35% end-to-end speedup using my fused GN kernel.

The CUDA kernels also compile about 5x faster than PyTorch's native GN CUDA kernels (from ~100 seconds to ~20 seconds) on my laptop (Intel i7-12700H @ 2.7 GHz CPU) as I removed many torch header files which slowed down compilation time.

## Todo
- [x] Working forward kernel for all size inputs 
- [x] Working backward kernel for all size inputs 
- [x] Forward/backward kernel surpasses PyTorch GN speed and matches stable-fast Triton GN
- [X] Python wrapper for fused forward activation (e.g. fused GN + Silu)
- [x] Backward pass with fused activation gradients
- [ ] TorchDynamo functionality?

## Other
### Can't compile extension
If you have problems compiling the extension, it may be because your Pytorch's CUDA version is incompatible with your nvcc version. To fix this, try upgrading/downgrading your PyTorch installation to match nvcc's version as close as possible or vice-versa.

To see your nvcc version: `nvcc -V`

See your PyTorch CUDA version: `python3 -c "import torch; print(torch.version.cuda)"`

### Writeup
I also wrote a writeup on the math behind the forward/backward pass for GN + fused activation. Check it out [here](https://latentcall145.github.io/gn-chronicles)!

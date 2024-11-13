import torch.nn.functional as F
import torch.nn as nn
import torch, os
module_dir = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load
gn_op = load(
        name="gn_op",
        sources=[
            os.path.join(module_dir, "csrc/custom_gn.cpp"),
            os.path.join(module_dir, "csrc/gn_kernel.cu"),
            ],
        extra_cuda_cflags=[
            '-use_fast_math',
            '-extra-device-vectorization',
            '-extended-lambda', # for gpu_kernel (although this isn't used in custom GN kernels)
            '-lineinfo', # useful for profiling
            '-src-in-ptx',
            ],
        extra_cflags=[
            '-Ofast', # needed or else GN NCHW from source is slower than nn.GroupNorm
            '-funroll-all-loops',
            '-march=native',
            ], 
        is_python_module=False,
        verbose=True,
        )

class GN_NHWC_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float, activation: str):
        ctx.x_shape = X.shape

        X_flat = X.view(X.shape[0], X.shape[1], -1)
        X_out, means, rstds = torch.ops.gnop.fwd(X_flat, weight, bias, G, eps, activation)
        ctx.save_for_backward(X_flat, weight, bias, means, rstds)
        ctx.G = G
        ctx.activation = activation
        return X_out.view(ctx.x_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        X_flat, weight, bias, means, rstds = ctx.saved_tensors 
        dy = dy.contiguous(memory_format=torch.channels_last).view(X_flat.shape)
        assert dy.stride() == X_flat.stride()
        dx, dgamma, dbeta = torch.ops.gnop.bwd(dy, X_flat, weight, bias, means, rstds, ctx.G, ctx.activation)
        return dx.view(ctx.x_shape), dgamma, dbeta, None, None, None

class GN_NHWC(nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, activation='identity', **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)
        activation_to_code = {
            'identity': 0,
            'relu': 1,
            'swish': 2,
            'silu': 2,
            'gelu': 3,
            'gelu_tanh': 4,
        }
        self.activation = activation
        self.act_code = activation_to_code[activation]

    @torch._dynamo.disable
    def forward(self, x):
        #N, C, H, W = x.shape
        #x = x.view(x.shape[0], x.shape[1], -1)
        G = self.num_groups
        if x.stride(1) == 1: # channels last format
            # make sure the other dims in x are contiguous (e.g. shape (2, 3, 5, 9) should have stride (135, 1, 27, 3) and not (135, 1, 3, 15))
            inner_dims = range(2, x.ndim)
            x_contiguous = x.permute(0, *inner_dims, 1).contiguous()
            inner_dims = range(1, x.ndim - 1)
            x = x_contiguous.permute(0, -1, *inner_dims)
            fwd_fn = GN_NHWC_Func.apply
        else: # channels first, fall back to torch's GN
            x = x.contiguous()
            activations = [lambda x: x, F.relu, F.silu, F.gelu, lambda x: F.gelu(x, approximate='tanh')]
            act_fn = activations[self.act_code]
            fwd_fn = lambda x, w, b, g, eps, _act: act_fn(F.group_norm(x, g, w, b, eps))

        if self.affine:
            return fwd_fn(x, self.weight, self.bias, self.num_groups, self.eps, self.act_code)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return fwd_fn(x, w, b, self.num_groups, self.eps, self.act_code)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}, activation={activation}'.format(**self.__dict__)

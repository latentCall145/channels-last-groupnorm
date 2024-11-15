# Checks if the outputs between NCHW and NHWC inputs are the same for the GN_NHWC layer
# note: this is not an exhaustive check
from gnnhwc import GN_NHWC
import torch.nn as nn
import torch

def test_nchw_nwhc():
    N, R, G, C = 1, 256, 32, 128
    x = torch.randn((N, C, R, R)).cuda()
    x_nhwc = x.to(memory_format=torch.channels_last)
    x.requires_grad_(True)
    x_nhwc.requires_grad_(True)
    rand_dy = torch.rand_like(x)
    rand_dy_nhwc = rand_dy.to(memory_format=torch.channels_last)

    for act in ['identity', 'silu', 'gelu', 'gelu_tanh']:
        for dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
            m = GN_NHWC(G, C, act).cuda().to(dtype)
            #m = nn.GroupNorm(G, C).cuda().to(dtype)
            out1 = m(x.to(dtype))
            out2 = m(x_nhwc.to(dtype))
            out1.backward(rand_dy)
            out2.backward(rand_dy_nhwc)
            assert (out1 - out2).square().mean() < 1e-6
            assert (x.grad - x_nhwc.grad).square().mean() < 1e-6

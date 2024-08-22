# Checks if the outputs between NCHW and NHWC inputs are the same for the GN_NHWC layer
# note: this is not an exhaustive check
from gnnhwc import GN_NHWC
import torch

def test_nchw_nwhc():
    N, R, G, C = 1, 256, 32, 128
    x = torch.randn((N, C, R, R)).cuda()
    x_nhwc = x.to(memory_format=torch.channels_last)

    for act in ['identity', 'silu', 'gelu', 'gelu_tanh']:
        for dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
            m = GN_NHWC(G, C, act).cuda().to(dtype)
            assert (m(x.to(dtype)) - m(x_nhwc.to(dtype))).square().mean() < 1e-6

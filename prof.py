# Runs the fwd + bwd pass of the GN_NHWC layer as well as nn.GroupNorm over a set of inputs.
# This script is meant to be run with Nsight Compute to evaluate kernel performance.
from gnnhwc import GN_NHWC
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch

if __name__ == '__main__':
    ACT_FN = 'identity'
    act_fn = {
        'identity': lambda x: x,
        'silu': F.silu,
        'relu': F.relu,
        'gelu': F.gelu,
        'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
    }[ACT_FN]

    inputs = [
        (torch.half, 8, 32, 512, 512, 32),
    ]

    for DTYPE, B, C, H, W, G in tqdm(inputs):
        print(f'profiling | DTYPE: {DTYPE} |B: {B:<2} | C: {C:<4} | H: {H:<4} | W: {W:<4} | G: {G:<3}')
        x = torch.empty((B, C, H, W), dtype=DTYPE, device='cuda')
        dy = torch.ones_like(x)
        x_nhwc = x.to(memory_format=torch.channels_last)
        dy_nhwc = dy.to(memory_format=torch.channels_last)
        x.requires_grad_(True)
        x_nhwc.requires_grad_(True)

        custom_gn = GN_NHWC(G, C, activation=ACT_FN).cuda().to(DTYPE)
        custom_gn(x_nhwc).backward(dy_nhwc)

        gn_ref_nhwc = nn.GroupNorm(G, C).cuda().to(DTYPE)
        gn_ref_nhwc(x_nhwc).backward(dy_nhwc)

        gn_ref_nchw = nn.GroupNorm(G, C).cuda().to(DTYPE)
        gn_ref_nchw(x).backward(dy)

# Tests if op can handle tensors of rank 3 or higher (GN normally uses rank 4 tensors but can take in shapes (N, C, *))
from gnnhwc import GN_NHWC
import torch.nn.functional as F
import torch.nn as nn
import torch

def test_trailing_dims():
    for N, C, R_list in [
        (1, 512, [1, 1, 1, 1]),
        (1, 256, [4, 2, 8, 1]),
        (1, 256, [7, 13, 19, 1]),
        (1, 128, [32, 1, 1, 1]),
        (1,  32, [32, 128, 32, 1]),

    ]:
        x = torch.randn((N, *R_list, C)).cuda()
        x = x.permute(0, -1, *range(1, len(R_list)+1)) # get channels-last-like behavior (namely x.stride(1) = 1) for tensors with ndim > 4
        G = 32
        for dtype in [torch.half, torch.float]:
            m = GN_NHWC(G, C).cuda().to(dtype)
            #m = nn.GroupNorm(G, C).cuda().to(dtype)
            m_ref = nn.GroupNorm(G, C).cuda().to(dtype)

            y = m(x.to(dtype))
            y_ref = m_ref(x.to(dtype))
            assert F.mse_loss(y, y_ref) < 1e-6 # make sure GN is consistent to the reference

            while len(R_list) != 1: # flattens out the input tensor one dim per iter, comparing it with the original tensor
                last_dim = R_list.pop(-1)
                R_list[-1] *= last_dim
                x = x.view(N, C, *R_list)
                y = m(x.to(dtype))
                assert F.mse_loss(y_ref.view(N, C, *R_list), y) < 1e-6

def test_weird_mem_fmt():
    B = 2
    C = 32
    G = 8
    '''
    test pseudo-channels last (x.stride(1) = 1 but non-channels-last) and pseudo-contiguous (x.stride(-1) = 1 but non-contiguous) inputs
    example: x.shape = (5, 4, 3, 2)
    contiguous:           x.stride = (24, 6, 2, 1)
    pseudo-contiguous:    x.stride = (24, 2, 8, 1)
    channels-last:        x.stride = (24, 1, 8, 4)
    pseudo-channels-last: x.stride = (24, 1, 4, 12)
    '''
    for dtype in [torch.half, torch.float]:
        m = GN_NHWC(G, C).cuda().to(dtype)
        #m = nn.GroupNorm(G, C).cuda().to(dtype)
        m_ref = nn.GroupNorm(G, C).cuda().to(dtype)

        for inner_dims in (
            (3, 5),
            (3, 5, 7),
            (3, 5, 7, 11),
        ):
            # pseudo-channels-last (x.stride(1) == 1)
            x = torch.randn((B, *inner_dims, C), device='cuda')
            x = x.permute(0, -1, *range(len(inner_dims), 0, -1))

            y = m(x.to(dtype))
            y_ref = m_ref(x.to(dtype))
            assert F.mse_loss(y, y_ref) < 1e-6 # make sure GN is consistent to the reference

            # pseudo-contiguous (x.stride(-1) == 1)
            x = torch.randn((B, C, *inner_dims), device='cuda')
            x = x.permute(0, 1, *range(len(inner_dims), 1, -1), -1)

            y = m(x.to(dtype))
            y_ref = m_ref(x.to(dtype))
            assert F.mse_loss(y, y_ref) < 1e-6 # make sure GN is consistent to the reference

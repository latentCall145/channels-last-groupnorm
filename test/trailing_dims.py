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
        x = x.view(N, -1, C).permute(0, 2, 1).view(N, C, *R_list) # hack to get channels_last behavior for non-4D tensors
        G = 32
        for dtype in [torch.half, torch.float]:
            m = GN_NHWC(G, C).cuda().to(dtype)
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

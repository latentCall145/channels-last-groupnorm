import torch.nn as nn
import numpy as np
import torch, datetime, time, os, itertools
torch.set_printoptions(sci_mode=False)
module_dir = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load
gn_op = load(
        name="gn_op",
        sources=[
            os.path.join(module_dir, "custom_gn.cpp"),
            os.path.join(module_dir, "N_grid_gn_kernel.cu"),
            os.path.join(module_dir, "NH_grid_gn_kernel.cu"),
            os.path.join(module_dir, "NG_grid_gn_kernel.cu"),
            os.path.join(module_dir, "fully_fused_gn_kernel.cu"),
            os.path.join(module_dir, "bwd_gn_kernel.cu"),
            os.path.join(module_dir, "nchw_kernel.cu")
            ],
        extra_cuda_cflags=[
            '-use_fast_math',
            '--extended-lambda', # necessary flag to use gpu_kernel in CUDA kernels
            '-lineinfo', # useful for profiling
            ],
        extra_cflags=['-O3'], # needed or else GN NCHW from source is slower than nn.GroupNorm
        verbose=True
        )

class GN_NHWC_Func(torch.autograd.Function):
    @staticmethod
    def choose_kernel(X: torch.Tensor, G: int):
        #return gn_op.fwd_NG_grid
        return gn_op.fwd_NH_grid
        #return gn_op.fwd_N_grid
        #if X.shape[0] <= 8: # and X.shape[2] * X.shape[3] >= 128 * 128: # and weight.dtype in (torch.bfloat16, torch.half):
        #    return gn_op.fwd_NH_grid
        #else:
        #    return gn_op.fwd_N_grid

    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float):
        fwd_fn = GN_NHWC_Func.choose_kernel(X, G)
        X_out, means, rstds = fwd_fn(X, weight, bias, G, eps)
        ctx.save_for_backward(X, weight, means, rstds, torch.Tensor([G]))
        return X_out

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous(memory_format=torch.channels_last)
        X, weight, means, rstds, G = ctx.saved_tensors 
        dx, dgamma, dbeta = gn_op.bwd(dy, X, weight, means, rstds, int(G))
        return dx, dgamma, dbeta, None, None

class GN_NHWC(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        #print(x.shape, self.num_channels)
        N, C, H, W = x.shape
        G = self.num_groups

        if C // G > 512:
            raise ValueError(f'Error in fwd for X.shape={x.shape}, G={G}: C // G = {C // G} which is greater than 512. This input is not supported.')

        f = max(1, C // 512)
        bdx = min(G // f, 512)
        d = min(H * G // f, 512) // bdx
        if W % d != 0:
            raise ValueError(f'Error in fwd for X.shape={x.shape}, G={G}: X has width {W} which is not a multiple of {d}. This input is not supported.')

        if self.affine:
            return GN_NHWC_Func.apply(x, self.weight, self.bias, self.num_groups, self.eps)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return GN_NHWC_Func.apply(x, w, b, self.num_groups, self.eps)

class GN_NCHW_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float):
        X_out, means, rstds = gn_op.nchwforward(X, weight, bias, G, eps)
        ctx.save_for_backward(X, weight, means, rstds, torch.Tensor([G]))
        return X_out

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        X, weight, means, rstds, G = ctx.saved_tensors 
        dx, dgamma, dbeta = gn_op.nchwbackward(dy, X, weight, means, rstds, int(G))
        return dx, dgamma, dbeta, None, None

class GN_NCHW(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        if self.affine:
            return GN_NCHW_Func.apply(x, self.weight, self.bias, self.num_groups, self.eps)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return GN_NCHW_Func.apply(x, w, b, self.num_groups, self.eps)

if __name__ == '__main__':
    DTYPE = torch.bfloat16
    print('DTYPE:', DTYPE)
    MODE = 'check' # can be 'check', 'bench', other modes do both

    if MODE != 'bench':
        B = 2
        C = 128
        R = 32
        G = 32
        x = torch.arange(B * C * R * R).reshape((B, C, R, R)).to(DTYPE, memory_format=torch.channels_last).cuda().requires_grad_(True) #* 100
        #torch.random.manual_seed(0)
        print(x)

        gn1 = nn.GroupNorm(G, C).cuda().to(DTYPE)
        #gn1 = GN_NCHW(G, C).cuda().to(DTYPE)
        gn2 = GN_NHWC(G, C).cuda().to(DTYPE)

        with torch.no_grad():
            w = torch.randn((C,))
            b = torch.randn((C,))
            gn1.weight.copy_(w)
            gn1.bias.copy_(b)
            gn2.weight.copy_(w)
            gn2.bias.copy_(b)
        g1 = gn1(x)
        g2 = gn2(x)
        print(g1)
        rand_dy = torch.rand_like(g2)
        rand_dy = torch.ones_like(g1)
        g1sum = (g1 * rand_dy).sum()
        g2sum = (g2 * rand_dy).sum()
        print('FORWARD')
        print('g2', g1.shape)
        print(g1-g2)

        print('BACKWARD')
        print('g1 sum wrt w')
        g1_grad_wrt_w = torch.autograd.grad(g1sum, gn1.weight, retain_graph=True)[0].reshape((gn1.weight.numel(),))
        g2_grad_wrt_w = torch.autograd.grad(g2sum, gn2.weight, retain_graph=True)[0].reshape((gn1.weight.numel(),))
        print(g1_grad_wrt_w - g2_grad_wrt_w)

        print('g1 sum wrt b')
        g1_grad_wrt_b = torch.autograd.grad(g1sum, gn1.bias, retain_graph=True)[0].reshape((gn1.bias.numel(),))
        g2_grad_wrt_b = torch.autograd.grad(g2sum, gn2.bias, retain_graph=True)[0].reshape((gn2.bias.numel(),))
        print(g1_grad_wrt_b - g2_grad_wrt_b)

        print('g1 sum wrt x')
        print('gt')
        g1_grad_wrt_x = torch.autograd.grad(g1sum, x, retain_graph=True)[0] #.reshape((x.numel(),))
        print(g1_grad_wrt_x)
        print('exp')
        g2_grad_wrt_x = torch.autograd.grad(g2sum, x, retain_graph=True)[0] #.reshape((x.numel(),))
        print(g2_grad_wrt_x)
        print('diff')
        print(g1_grad_wrt_x-g2_grad_wrt_x)

    if MODE != 'check':
        NSEC = 5 # number of seconds that each kernel runs for on a certain input
        BATCHES = [1, 2, 4, 8, 16, 32]
        CHANNELS = [32, 64, 128, 256, 512]
        RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256, 512]
        NUM_GROUPS = [4, 8, 16, 32, 64, 128]
        GN_KERNELS = [
                (GN_NHWC, 'GN NHWC fused (custom op)', gn_op.fwd_fused),
                (GN_NHWC, 'GN NHWC NH grid (custom op)', gn_op.fwd_NH_grid),
                (GN_NHWC, 'GN NHWC N grid (custom op)', gn_op.fwd_N_grid),
                (GN_NHWC, 'GN NHWC NG grid (custom op)', gn_op.fwd_NG_grid),
                (GN_NCHW, 'torch.nn GN NCHW (compiled from src)', gn_op.nchwforward),
        ]

        os.makedirs('csvs', exist_ok=True)
        fname = datetime.datetime.now().strftime("csvs/%H-%M-%S-%d-%m-%Y.csv")
        print(f'Writing to {fname}')
        outfile = open(fname, 'w')
        outfile.write('Kernel,B (batch),C (num channels),R (resolution),G (num groups), D (C/G),Speed (it/s; 25th percentile),Speed (it/s; 50th percentile),Speed (it/s; 75th percentile)\n')

        def config_filter(x): # returns true if config is valid
            B, C, R, G = x
            if G > C:
                return False

            dtype_size = 2 if DTYPE in (torch.half, torch.bfloat16) else 4 # only care about 16/32-bit dtypes for now
            estimated_mem_usage_gib = (3 * dtype_size * B * C * R * R) / 2**30 # main VRAM tensors: X_nchw (shape=(B,C,R,R)), X_nhwc (same shape), Y (same shape)
            if estimated_mem_usage_gib > 4: # vram filter
                return False
            return True
        
        configs = list(filter(config_filter, itertools.product(BATCHES, CHANNELS, RESOLUTIONS, NUM_GROUPS)))
        print('Estimated time (seconds) to complete:', NSEC * len(configs) * len(GN_KERNELS))

        for B, C, R, G in configs:
            x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda').requires_grad_(True)
            x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).cuda().requires_grad_(True)

            gn_args = (G, C)
            BENCH = 'fwd' # can be 'fwd', 'bwd', anything else is fwd + bwd
            print(BENCH, 'X shape:', x_nchw.shape, 'G (num groups):', G)
            for gn_class, desc, fwd_fn in GN_KERNELS:
                gn_input = x_nchw if 'NCHW' in desc else x_nhwc
                print(desc)

                try:
                    gn_layer = gn_class(*gn_args).cuda().to(DTYPE)
                    g = gn_layer(gn_input)
                    torch.cuda.synchronize()

                    tic = time.time()
                    tic_sec = time.time()
                    ntrials = 0
                    ntrials_minor = 0
                    minor_speeds = [] # used to track speed percentiles since they can often vary by a lot

                    while time.time() - tic < NSEC:
                        if BENCH != 'bwd':
                            g = fwd_fn(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps) # Not calling gn_layer(gn_input) since I found this added a lot of overhead
                        if BENCH != 'fwd':
                            torch.autograd.grad(g_mem_fmt.sum(), gn_input, retain_graph=True)
                        torch.cuda.synchronize()

                        ntrials += 1
                        ntrials_minor += 1

                        if time.time() - tic_sec > 0.1:
                            speed = round(ntrials_minor / (time.time() - tic_sec), 2)
                            minor_speeds.append(speed)
                            print(f'{round(time.time() - tic, 1)}/{NSEC} seconds completed, speed: {speed} it/s\r', end='')
                            ntrials_minor = 0
                            tic_sec = time.time()

                    minor_speeds = np.array(minor_speeds)
                    median_speed = round(np.percentile(minor_speeds, 50), 2)
                    slow_speed = round(np.percentile(minor_speeds, 25), 2)
                    fast_speed = round(np.percentile(minor_speeds, 75), 2)
                    print(f'\nSpeed (25th/50th/75th percentile): {slow_speed}/{median_speed}/{fast_speed} it/s')
                except KeyboardInterrupt:
                    print(f'Keyboard interrupt, closing {fname}.')
                    outfile.close()
                    raise
                except Exception as e:
                    #print('Error:', e)
                    median_speed = slow_speed = fast_speed = '-1 (failed)'
                    print(f'FAILED')
                
                outfile.write(f'{desc},{B},{C},{R},{G},{C//G},{slow_speed},{median_speed},{fast_speed}\n')
            print()
        print(f'All tests done, closing {fname}.')
        outfile.close()

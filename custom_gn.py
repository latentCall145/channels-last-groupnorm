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
            os.path.join(module_dir, "NH_grid_gn_kernel.cu"),
            os.path.join(module_dir, "bwd_gn_kernel.cu"),
            os.path.join(module_dir, "nchw_kernel.cu")
            ],
        extra_cuda_cflags=[
            '-use_fast_math',
            '--extended-lambda', # necessary flag to use gpu_kernel in CUDA kernels
            '-lineinfo', # useful for profiling
            ],
        extra_cflags=[
            '-O3', # needed or else GN NCHW from source is slower than nn.GroupNorm
            '-funroll-all-loops',
            '-march=native',
            ], 
        verbose=True
        )

class GN_NHWC_Func(torch.autograd.Function):
    @staticmethod
    def choose_kernel(X: torch.Tensor, G: int):
        return gn_op.fwd_NH_grid
        #if X.shape[1] / G * X.shape[2] * X.shape[3] <= 8192: # and weight.dtype in (torch.bfloat16, torch.half):
        #    #print('fs', X.shape)
        #    return gn_op.fwd_fused
        #else:
        #    #print('NH', X.shape)
        #    return gn_op.fwd_NH_grid

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

        #if H * W % 8 != 0:
        #    raise ValueError(f'Error in fwd for X.shape={x.shape}, G={G}: H * W is not a multiple of 8. This input is not supported.')

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
    DTYPE = torch.float
    DTYPE = torch.double
    print('DTYPE:', DTYPE)
    MODE = 'check' # can be 'check', 'bench', other modes do both
    CHECK_PROF = False

    if MODE != 'bench':
        #DTYPEs = (torch.bfloat16, torch.float, torch.double)
        #Bs = (2, 2, 4, 8, 16)
        #Rs = (8, 16, 64, 256, 512)
        #Cs = (32, 64, 128, 256, 512)
        #itertools.product(DTYPEs, Bs, Rs, Cs)

        for B, R, C, G in (
            (1, 64, 960, 32),
            (1, 64, 640, 32),
            (1, 64, 256, 32),
            (1, 32, 1920, 32),
            (2, 32, 1280, 32),
            (1, 64, 320, 32),
            (1, 32, 960, 32),
            (1, 16, 2560, 32),
            (1, 32, 640, 32),
            (1, 16, 1920, 32),
            (1, 16, 1280, 32),
            (1, 32, 320, 32),
            (1, 8, 2560, 32),
            (1, 16, 640, 32),
            (1, 8, 1280, 32),

            (1, 512, 64, 32),
            (1, 64, 512, 32),
            (1, 64, 256, 32),
            (2, 64, 128, 32),
            (2, 32, 128, 32),
            (2, 8, 512, 32),
            (2, 4, 512, 32),
            (1, 4, 512, 32),
            (2, 4, 256, 32),
            (1, 4, 256, 32),
            (13, 65, 961, 31),
            (2, 65, 128, 32),
            (1, 3, 6, 2),
        ):
            dtype_size = 2 if DTYPE in (torch.half, torch.bfloat16) else 4 # only care about 16/32-bit dtypes for now
            estimated_mem_usage_gib = 2 * (4.5 * dtype_size * B * C * R * R) / 2**30 # main VRAM tensors: X_nchw (shape=(B,C,R,R)), X_nhwc (same shape), Y (same shape)
            if estimated_mem_usage_gib > 4: # vram filter
                continue
            torch.cuda.empty_cache()
            print(f'B: {B:<2} | C: {C:<4} | R: {R:<4} | G: {G:<3} | DTYPE: {DTYPE}')
            x = torch.randn(B * C * R * R).reshape((B, C, R, R)).to(DTYPE, memory_format=torch.channels_last).cuda().requires_grad_(True) #* 1000
            torch.random.manual_seed(0)

            gn2 = GN_NHWC(G, C).cuda().to(DTYPE)

            if CHECK_PROF:
                #g1 = gn1(x.contiguous())
                #g1sum = g1.sum()
                #g1_grad_wrt_w = torch.autograd.grad(g1sum, gn1.weight, retain_graph=True)[0]
                g2 = gn2(x)
                g2sum = g2.sum()
                g2_grad_wrt_w = torch.autograd.grad(g2sum, gn2.weight, retain_graph=True)[0]
            else:
                gn1 = nn.GroupNorm(G, C).cuda().to(DTYPE)
                with torch.no_grad():
                    w = torch.randn((C,)) #* 1000
                    b = torch.randn((C,)) #* 1000
                    gn1.weight.copy_(w)
                    gn1.bias.copy_(b)
                    gn2.weight.copy_(w)
                    gn2.bias.copy_(b)
                g1 = gn1(x.contiguous())
                g2 = gn2(x)
                rand_dy = torch.rand_like(g2)
                g1sum = (g1 * rand_dy).sum()
                g2sum = (g2 * rand_dy).sum()

                def print_err(act1, act2, left_pad=0):
                    lpad = ' ' * left_pad
                    if (act1 - act2).abs().mean() > 1e-5:
                        print(f'{lpad}{act1 - act2}')
                        print(f'{lpad}{(act1 - act2).abs().mean()}')
                    else:
                        print(f'{lpad}No difference found')

                print('  FORWARD')
                print_err(g1, g2, 4)

                print('  BACKWARD')
                print('    wrt weight')
                g1_grad_wrt_w = torch.autograd.grad(g1sum, gn1.weight, retain_graph=True)[0]
                g2_grad_wrt_w = torch.autograd.grad(g2sum, gn2.weight, retain_graph=True)[0]
                print_err(g1_grad_wrt_w, g2_grad_wrt_w, 6)

                print('    wrt bias')
                g1_grad_wrt_b = torch.autograd.grad(g1sum, gn1.bias, retain_graph=True)[0].reshape((gn1.bias.numel(),))
                g2_grad_wrt_b = torch.autograd.grad(g2sum, gn2.bias, retain_graph=True)[0].reshape((gn2.bias.numel(),))
                print_err(g1_grad_wrt_b, g2_grad_wrt_b, 6)

                print('    wrt X')
                g1_grad_wrt_x = torch.autograd.grad(g1sum, x, retain_graph=True)[0] #.reshape((x.numel(),))
                g2_grad_wrt_x = torch.autograd.grad(g2sum, x, retain_graph=True)[0] #.reshape((x.numel(),))
                print_err(g1_grad_wrt_x, g2_grad_wrt_x, 6)

    if MODE != 'check':
        NSEC = 1 # number of seconds that each kernel runs for on a certain input
        BATCHES = [1, 2, 4, 8, 16, 32]
        #CHANNELS = [32, 64, 128, 256, 512]
        CHANNELS = [320, 640, 960, 1920, 2560]
        RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256, 512]
        #NUM_GROUPS = [4, 8, 16, 32, 64, 128]
        NUM_GROUPS = [32]
        BENCH = 'fwd' # can be 'fwd', 'bwd', anything else is fwd + bwd
        GN_KERNELS = [
                #(GN_NHWC, 'GN NHWC fused (custom op)', gn_op.fwd_fused),
                #(GN_NHWC, 'GN NHWC NH grid (custom op)', gn_op.fwd_NH_grid),
                #(GN_NHWC, 'GN NHWC N grid (custom op)', gn_op.fwd_N_grid),
                #(GN_NHWC, 'GN NHWC NG grid NG grid (custom op)', gn_op.fwd_NG_grid),
                #(GN_NCHW, 'torch.nn GN NCHW (compiled from src)', gn_op.nchwforward),
                #(nn.GroupNorm, 'torch.nn GN NCHW', None),
                (GN_NCHW, 'torch.nn GN NCHW (compiled from src)', None),
                (GN_NHWC, 'GN NHWC', None),
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

            '''
            return (B, C, R, G) in {
                    (1,  512, 8, 32),
                    #(2,  512, 8, 32),
                    #(4,  512, 8, 32),
                    #(8,  512, 8, 32),
                    (16, 512, 8, 32),
                    #(32, 512, 8, 32),

                    (1,  512, 16, 32),
                    #(2,  512, 16, 32),
                    #(4,  512, 16, 32),
                    #(8,  512, 16, 32),
                    (16, 512, 16, 32),
                    #(32, 512, 16, 32),

                    (1, 64, 256, 16),
                    #(2, 64, 256, 16),
                    #(4, 64, 256, 16),
                    #(8, 64, 256, 16),
                    (16, 64, 256, 16),
                    #(32, 64, 256, 16),

                    (1, 64, 512, 16),
                    #(2, 64, 512, 16),
                    #(4, 64, 512, 16),
                    #(8, 64, 512, 16),
                    (4, 64, 512, 16),
                    #(32, 64, 512, 16),

                    #(1, 128, 64, 16),
                    #(2, 128, 64, 16),
                    #(4, 128, 32, 16),
                    #(8, 128, 32, 16),
                    #(16, 128, 32, 16),
                    #(32, 128, 32, 16),
                    }
            '''

            dtype_size = 2 if DTYPE in (torch.half, torch.bfloat16) else 4 # only care about 16/32-bit dtypes for now
            estimated_mem_usage_gib = (3 * dtype_size * B * C * R * R) / 2**30 # main VRAM tensors: X_nchw (shape=(B,C,R,R)), X_nhwc (same shape), Y (same shape)
            if estimated_mem_usage_gib > 4: # vram filter
                return False
            return True
        
        configs = list(filter(config_filter, itertools.product(BATCHES, CHANNELS, RESOLUTIONS, NUM_GROUPS)))
        print('Estimated time (seconds) to complete:', NSEC * len(configs) * len(GN_KERNELS))

        def red(text):
            return '\033[91m' + str(text) + '\033[0m'
        def green(text):
            return '\033[92m' + str(text) + '\033[0m'
        def yellow(text):
            return '\033[93m' + str(text) + '\033[0m'
        def blue(text):
            return '\033[94m' + str(text) + '\033[0m'

        for B, C, R, G in configs:
            x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda').requires_grad_(True)
            x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).cuda().requires_grad_(True)

            gn_args = (G, C)
            print(BENCH, 'X shape:', x_nchw.shape, 'G (num groups):', G)
            for gn_class, desc, fwd_fn in GN_KERNELS:
                gn_input = x_nchw if 'NCHW' in desc else x_nhwc
                print(f'\t{desc}')

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
                        if BENCH == 'fwd':
                            if fwd_fn is None:
                                g = gn_layer(gn_input)
                            else:
                                g = fwd_fn(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps) # Not calling gn_layer(gn_input) since I found this added a lot of overhead
                        elif BENCH == 'both':
                            g = gn_layer(gn_input)
                        if BENCH != 'fwd':
                            torch.autograd.grad(g.sum(), gn_input, retain_graph=True)
                        torch.cuda.synchronize()

                        ntrials += 1
                        ntrials_minor += 1

                        if time.time() - tic_sec > 0.1:
                            speed = round(ntrials_minor / (time.time() - tic_sec), 2)
                            minor_speeds.append(speed)
                            print(f'\t\t{round(time.time() - tic, 1)}/{NSEC} seconds completed, speed: {blue(speed)} it/s\r', end='')
                            ntrials_minor = 0
                            tic_sec = time.time()

                    minor_speeds = np.array(minor_speeds)
                    median_speed = round(np.percentile(minor_speeds, 50), 2)
                    slow_speed = round(np.percentile(minor_speeds, 25), 2)
                    fast_speed = round(np.percentile(minor_speeds, 75), 2)
                    print(f'\n\t\tSpeed (25th/50th/75th percentile): {red(slow_speed)}/{yellow(median_speed)}/{green(fast_speed)} it/s')
                except KeyboardInterrupt:
                    print(f'Keyboard interrupt, closing {fname}.')
                    outfile.close()
                    raise
                except Exception as e:
                    print('\t\tFAILED; Error:', str(e).strip())
                    median_speed = slow_speed = fast_speed = '-1 (failed)'
                
                outfile.write(f'{desc},{B},{C},{R},{G},{C//G},{slow_speed},{median_speed},{fast_speed}\n')
            print()
        print(f'All tests done, closing {fname}.')
        outfile.close()

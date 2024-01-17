import torch.nn as nn
import torch
import datetime, time, os, itertools
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
        return gn_op.fwd_NH_grid
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
        dx, dgamma, dbeta = gn_op.backward(dy, X, weight, means, rstds, int(G))
        return dx, dgamma, dbeta, None, None

class GN_NHWC(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        #print(x.shape, self.num_channels)
        N, C, H, W = x.shape
        G = self.num_groups
        f = max(1, C // 512)
        bdx = min(G // f, 512)
        d = min(H * G, 512) // bdx
        if C // G > 512:
            raise ValueError(f'C // G = {C // G} which is greater than 512. This input is not supported.')
        if W % d != 0:
            raise ValueError(f'X[0] has width {W} which is not a multiple of {d}. This input is not supported.')
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
    DTYPE = torch.float
    print('DTYPE:', DTYPE)
    MODE = 'bench' # can be 'check', 'bench', other modes do both

    if MODE != 'bench':
        B = 4
        C = 256
        R = 8
        G = 32
        x = torch.arange(B * C * R * R).reshape((B, C, R, R)).to(DTYPE, memory_format=torch.channels_last).cuda().requires_grad_(True) #* 100
        #torch.random.manual_seed(0)

        gn1 = nn.GroupNorm(G, C).cuda().to(DTYPE)
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
        print('FORWARD')
        print('g2', g1.shape)
        print(g1-g2)

        #print('BACKWARD')
        #print('g1 sum wrt x')
        #g1_grad_wrt_x = torch.autograd.grad(g1.sum(), x, retain_graph=True)[0] #.reshape((x.numel(),))
        #g2_grad_wrt_x = torch.autograd.grad(g2.sum(), x, retain_graph=True)[0] #.reshape((x.numel(),))
        #print(g1_grad_wrt_x-g2_grad_wrt_x)

        #print('g1 sum wrt w')
        #g1_grad_wrt_w = torch.autograd.grad(g1.sum(), gn1.weight, retain_graph=True)[0].reshape((gn1.weight.numel(),))
        #g2_grad_wrt_w = torch.autograd.grad(g2.sum(), gn2.weight, retain_graph=True)[0].reshape((gn1.weight.numel(),))
        #print(g1_grad_wrt_w - g2_grad_wrt_w)

        #print('g1 sum wrt b')
        #g1_grad_wrt_b = torch.autograd.grad(g1.sum(), gn1.bias, retain_graph=True)[0].reshape((gn1.bias.numel(),))
        #g2_grad_wrt_b = torch.autograd.grad(g2.sum(), gn2.bias, retain_graph=True)[0].reshape((gn2.bias.numel(),))
        #print(g1_grad_wrt_b - g2_grad_wrt_b)

    NSEC = 1 # number of seconds that each kernel runs for on a certain input
    outfile = open(datetime.datetime.now().strftime("%H-%M-%S-%d-%m-%Y.csv"), 'w')
    outfile.write('Kernel,B (batch),C (num channels),R (resolution),G (num groups), D (C/G),Speed (it/s)\n')
    if MODE != 'check':
        for B, C, R, G in itertools.product(
                [1, 2, 4, 8, 16, 32],
                [32, 64, 128, 256, 512],
                [4, 8, 16, 32, 64, 128, 256, 512],
                [2, 4, 8, 16, 32, 64, 128]
                ):
            if B * C * R * R > 8 * 64 * 256 * 256:
                continue
            if G > C:
                continue

            x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda').requires_grad_(True)
            x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).cuda().requires_grad_(True)
            gn_args = (G, C)
            BENCH = 'fwd' # can be 'fwd', 'bwd', anything else is fwd + bwd
            print(BENCH, 'X shape:', x_nchw.shape, 'G (num groups):', G)
            for gn_class, gn_input, desc, fwd_fn in (
                    (GN_NHWC, x_nhwc, 'GN NHWC NH grid (custom op)', gn_op.fwd_NH_grid),
                    (GN_NHWC, x_nhwc, 'GN NHWC N grid (custom op)', gn_op.fwd_N_grid),
                    (GN_NHWC, x_nhwc, 'GN NHWC NG grid (custom op)', gn_op.fwd_NG_grid),
                    (GN_NCHW, x_nchw, 'torch.nn GN NCHW (compiled from src)', None),
                    ):
                print(desc)

                gn_layer = gn_class(*gn_args).cuda().to(DTYPE)

                # Not calling gn_layer(gn_input) since I found this added a lot of overhead
                if isinstance(gn_layer, GN_NHWC):
                    g = fwd_fn(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps)
                elif isinstance(gn_layer, GN_NCHW):
                    g = gn_op.nchwforward(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps)
                else:
                    g = gn_layer(gn_input)
                torch.cuda.synchronize()

                tic = time.time()
                tic_sec = time.time()
                ntrials = 0
                while time.time() - tic < NSEC:
                    if BENCH != 'bwd':
                        if isinstance(gn_layer, GN_NHWC):
                            g = fwd_fn(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps)
                        elif isinstance(gn_layer, GN_NCHW):
                            g = gn_op.nchwforward(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps)
                        else:
                            g = gn_layer(gn_input)
                    if BENCH != 'fwd':
                        if 'NHWC' in desc:
                            g_mem_fmt = g.contiguous(memory_format=torch.channels_last) # in NHWC models, must convert possibly NCHW outputs into NHWC (i.e. from nn GN), note that this is a no-op if g is already in NHWC format (e.g. GN_NHWC output)
                        else:
                            g_mem_fmt = g.contiguous()
                        torch.autograd.grad(g_mem_fmt.sum(), gn_input, retain_graph=True)
                    torch.cuda.synchronize()
                    ntrials += 1

                    if time.time() - tic_sec > 1:
                        speed = round(ntrials / (time.time() - tic), 2)
                        print(f'{round(time.time() - tic, 1)}/{NSEC} seconds completed, speed: {speed} it/s\r', end='')
                        tic_sec = time.time()
                speed = round(ntrials / NSEC, 2)
                print(f'\nSpeed: {speed} it/s')
                outfile.write(f'{desc},{B},{C},{R},{G},{C//G},{speed}\n')
            print()
        outfile.close()

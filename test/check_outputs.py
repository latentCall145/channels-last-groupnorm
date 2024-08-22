from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, datetime, time, os, itertools, sys
torch.set_printoptions(sci_mode=False, edgeitems=1)

class GN_Naive(nn.Module):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__()
        self.G = num_groups
        self.C = nc
        self.eps = 1e-05
        self.weight = nn.Parameter(torch.ones((nc,)))
        self.bias = nn.Parameter(torch.zeros((nc,)))
        self.x = None
        self.xnorm = None
        self.means = self.rstds = None

    def forward(self, x):
        N, C, H, W = x.shape
        self.x = x
        xr = x.view(N, self.G, H*W*C//self.G)
        means = xr.mean(dim=2, keepdim=True)
        rstds = torch.rsqrt(xr.var(dim=2, correction=0, keepdim=True) + self.eps)
        self.means = means[:, :, 0]
        self.rstds = rstds[:, :, 0]
        xnorm = (xr - means) * rstds
        self.xnorm = xnorm
        xnormr = xnorm.reshape(N, C, H, W) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return xnormr

    def bwd(self, dy): # bwd pass for debugging
        N, C, H, W = dy.shape
        G = self.G
        D = C // G
        dyr = dy.view(N, G, D, H, W)
        xr = x.view(N, G, D, H, W)
        dy_sum = dyr.sum((3, 4))
        xdy_sum = (dyr * xr).sum((3, 4))
        dy_gamma = (self.weight.view(1, G, D) * dy_sum).sum(2)
        xdy_gamma = (self.weight.view(1, G, D) * xdy_sum).sum(2)
        dweight = ((xdy_sum - self.means[:,:,None] * dy_sum) * self.rstds[:,:,None]).sum(0)
        dbias = dy_sum.sum(0)
        c1 = (self.means * dy_gamma - xdy_gamma) / (H*W*D) * self.rstds**3
        c2 = -self.means * c1 - dy_gamma*self.rstds / (H*W*D)
        dx = self.weight.view(1,G,D,1,1)*self.rstds.view(N,G,1,1,1)*dyr + c1.view(N,G,1,1,1)*xr+c2.view(N,G,1,1,1)

def red(text): return '\033[91m' + str(text) + '\033[0m'
def green(text): return '\033[92m' + str(text) + '\033[0m'
def yellow(text): return '\033[93m' + str(text) + '\033[0m'
def blue(text): return '\033[94m' + str(text) + '\033[0m'

def config_filter(x): # returns true if config is valid
    DTYPE, B, C, R, G = x
    if C % G != 0:
        return False
    if R == 1: # this causes an autograd problem where it gets confused since the tensor is both contiguous in NCHW/NHWC format 
        return False

    dtype_size = {torch.half: 2, torch.bfloat16: 2, torch.float: 4, torch.double: 8}[DTYPE] # only care about 16/32-bit dtypes for now
    #estimated_mem_usage_gib = (25 * dtype_size * B * C * R * R) / 2**30 #  this is just a rough estimate, likely wrong
    estimated_mem_usage_gib = (5 * dtype_size * B * C * R * R) / 2**30 #  this is just a rough estimate, likely wrong
    if estimated_mem_usage_gib > 3: # vram filter
        return False
    return True

bigx = None
def check_params(params, verbose=True):
    global bigx
    if bigx is None:
        bigx = torch.randn(128*1024*1024)
    vprint = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None
    DTYPE, B, C, R, G = params
    vprint(blue(f'output testing | DTYPE: {DTYPE} |B: {B:<2} | C: {C:<4} | R: {R:<4} | G: {G:<3}'))
    xc = bigx[:B*C*R*R].reshape((B, C, R, R)).to(DTYPE).cuda()
    x = xc.to(memory_format=torch.channels_last)
    xc.requires_grad_(True)
    x.requires_grad_(True)
    torch.random.manual_seed(0)

    gn2 = GN_NHWC(G, C, activation=ACT_FN).cuda().to(DTYPE)

    if CHECK_PROF:
        gn2(x).sum().backward()
        gn1 = nn.GroupNorm(G, C).cuda().to(DTYPE)
        gn1(x.contiguous()).sum().backward()
    else:
        gn1 = nn.GroupNorm(G, C).cuda().to(DTYPE)
        gnref = GN_Naive(G, C).cuda().to(DTYPE)
        with torch.no_grad():
            w = torch.randn((C,), dtype=DTYPE)
            b = torch.randn((C,), dtype=DTYPE)
            gn1.weight.copy_(w.detach().float())
            gn1.bias.copy_(b.detach().float())
            gn2.weight.copy_(w.detach())
            gn2.bias.copy_(b.detach())
            gnref.weight.copy_(w.detach())
            gnref.bias.copy_(b.detach())

        gn_layers = [gn1, gn2, gnref]
        g1 = act_fn(gn1(xc))
        g2 = gn2(x)
        rand_dy = torch.rand_like(g2)
        rand_dy /= rand_dy.numel() ** 0.5 # to prevent false positive errors from ocurring because of really large magnitude losses

        ERRS = {
            torch.bfloat16: 1e-6,
            torch.float16: 1e-7,
            torch.float: 1e-10,
            torch.double: 1e-20,
        }

        err_params = False
        gref = gref_dx = None
        def vprint_err(x_ref, x_test, x_naive_fn, bwd=True, left_pad=0):
            nonlocal gref, gref_dx
            lpad = ' ' * left_pad
            with torch.no_grad():
                err = F.mse_loss(x_ref, x_test)

            if err < ERRS[DTYPE]:
                vprint(green(f'{lpad}Negligible difference (err: {err:.2e}) found'))
            else:
                if gref is None:
                    gref = act_fn(gnref(xc))
                if bwd and gref_dx is None:
                    xc.grad = None
                    (gref * rand_dy).sum().backward()
                    gref_dx = xc.grad.clone()

                with torch.no_grad():
                    x_naive = x_naive_fn()
                    err_ref_naive = F.mse_loss(x_ref, x_naive)
                    err_test_naive = F.mse_loss(x_test, x_naive)

                if err_test_naive < err_ref_naive:
                    vprint(yellow(f'{lpad}Negligible difference (err: {err:.2e}, test-naive: {err_test_naive:.2e}, ref-naive: {err_ref_naive:.2e}) found'))
                else:
                    vprint(red(f'{lpad}Error: {err:.2e}, test-naive: {err_test_naive:.2e}, ref-naive: {err_ref_naive:.2e}'))
                    return True
            return False

        vprint('  FORWARD')
        err_params = vprint_err(g1, g2, lambda: gref, bwd=False, left_pad=4) or err_params
        vprint('  BACKWARD')
        xc.grad = None
        (g1 * rand_dy).sum().backward()
        g1_dx = xc.grad.clone()

        x.grad = None
        (g2 * rand_dy).sum().backward()
        g2_dx = x.grad.clone()

        vprint('    wrt X')
        err_params = vprint_err(g1_dx, g2_dx, lambda: gref_dx, left_pad=6) or err_params
        vprint('    wrt weight')
        err_params = vprint_err(gn1.weight.grad, gn2.weight.grad, lambda: gnref.weight.grad, left_pad=6) or err_params
        vprint('    wrt bias')
        err_params = vprint_err(gn1.bias.grad, gn2.bias.grad, lambda: gnref.bias.grad, left_pad=6) or err_params

        return err_params

if __name__ == '__main__':
    ACT_FN = 'silu'
    act_fn = {
        'identity': lambda x: x,
        'silu': F.silu,
        'relu': F.relu,
        'gelu': F.gelu,
        'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
    }[ACT_FN]

    MODE = 'check' # can be 'check', 'bench', other modes do both
    CHECK_PROF = len(sys.argv) > 1 and sys.argv[1] == '1'

    if MODE != 'bench':
        Bs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16)
        Cs = (
            1, 2, 3, 4, 5, 6, 7, 32, 64, 128, 256, 512,
            13, 140, 125, 961,
            160, 320, 640, 960, 1280, 1600, 1920, 2240, 2560, 1303, 2602, 3909
        )
        Rs = (
            2, 3, 4, 5, 6, 7, 8, 9, 10, 17,
            8, 16, 64, 128, 256, 512,
            1024,
        )
        Gs = (1, 2, 3, 4, 8, 16, 32,)
        all_params = itertools.product([torch.float], Bs, Cs, Rs, Gs)

        inputs = [
            #(torch.double, 1, 1, 2, 1),

            #(torch.double, 2, 3909, 5, 3),
            #(torch.double, 1, 2062, 5, 1031),
            #(torch.double, 3, 4096, 7, 4),
            #(torch.double, 1, 4096, 7, 4),

            #(torch.double, 2, 160, 8, 160),
            #(torch.double, 1, 3, 7, 1),
            #(torch.double, 1, 1, 4, 1),
            #(torch.double, 1, 128, 8, 8),
            #(torch.double, 2, 1280, 8, 32),
            #(torch.double, 2, 640, 16, 32),
            #(torch.double, 2, 2560, 8, 32),
            #(torch.double, 2, 1280, 16, 32),
            #(torch.double, 2, 320, 32, 32),
            #(torch.double, 2, 1920, 16, 32),
            #(torch.double, 2, 2560, 16, 32),
            #(torch.double, 2, 640, 32, 32),
            #(torch.double, 2, 960, 32, 32),
            #(torch.double, 2, 1280, 32, 32),
            #(torch.double, 2, 320, 64, 32),
            #(torch.double, 2, 1920, 32, 32),
            #(torch.double, 2, 640, 64, 32),
            #(torch.float, 1, 64, 256, 32),
            #(torch.float, 2, 64, 512, 32),
            (torch.double, 8, 128, 64, 32),
        ]
        inputs = None

        err_inputs = filter(config_filter, all_params)

        if inputs is None: # run on cartesian product of inputs
            for DTYPE in [torch.float, torch.double]: # run tests on low-precision dtypes and rerunning failing tests on higher-precision dtypes to see if there's an actual problem in the code or just a precision error
                inputs = [(DTYPE, *params[1:]) for params in err_inputs]
                err_inputs = []
                for params in tqdm(sorted(
                    inputs,
                    key = lambda x: x[1]*x[2]*x[3]*x[4]
                )):
                    err_params = check_params(params)
                    if err_params:
                        err_inputs.append(params)
        else:
            err_inputs = []
            for params in tqdm(inputs):
                err_params = check_params(params)
                if err_params:
                    err_inputs.append(params)

        if len(err_inputs) > 0:
            print(red('Error inputs found:'))
            print(err_inputs)
        elif not CHECK_PROF:
            print(green('No errors found :)'))

    if MODE != 'check':
        NSEC = 1 # number of seconds that each kernel runs for on a certain input
        DTYPES = [torch.bfloat16]
        #BATCHES = [1, 2, 4, 8, 16, 32]
        #CHANNELS = [32, 64, 128]
        #RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256, 512]
        #NUM_GROUPS = [32]

        BATCHES = [1, 8]
        CHANNELS = [32, 128]
        RESOLUTIONS = [64, 512]
        NUM_GROUPS = [32]

        BENCH = 'bwd' # can be 'fwd', anything else is fwd + bwd
        GN_KERNELS = [
                #(GN_NCHW, 'torch.nn GN NCHW (compiled from src)'),
                (nn.GroupNorm, 'torch.nn GN NCHW'),
                (GN_NHWC, 'GN NHWC'),
        ]

        os.makedirs('csvs', exist_ok=True)
        fname = datetime.datetime.now().strftime("csvs/%H-%M-%S-%d-%m-%Y.csv")
        print(f'Writing to {fname}')
        outfile = open(fname, 'w')
        outfile.write('Kernel,B (batch),C (num channels),R (resolution),G (num groups), D (C/G),Speed (it/s; 25th percentile),Speed (it/s; 50th percentile),Speed (it/s; 75th percentile)\n')
        
        configs = list(filter(config_filter, itertools.product(DTYPES, BATCHES, CHANNELS, RESOLUTIONS, NUM_GROUPS)))
        print('Estimated time (seconds) to complete:', NSEC * len(configs) * len(GN_KERNELS))

        for DTYPE, B, C, R, G in configs:
            x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda', requires_grad=True)
            x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).detach().requires_grad_(True)

            gn_args = (G, C)
            print(blue(f'benchmark ({BENCH}) | DTYPE: {DTYPE} | B: {B} | C: {C} | R: {R} | G: {G}'))
            for gn_class, desc in GN_KERNELS:
                gn_input = x_nchw if 'NCHW' in desc else x_nhwc
                grad = torch.ones_like(gn_input)
                print(f'\t{desc}')

                try:
                    gn_layer = gn_class(*gn_args).to(DTYPE).cuda()

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        g = gn_layer(gn_input)
                        if not isinstance(gn_layer, GN_NHWC):
                            g = act_fn(g)
                        if BENCH != 'fwd':
                            g.backward(grad)
                    torch.cuda.synchronize()

                    tic = time.time()
                    tic_sec = time.time()
                    ntrials = ntrials_minor = 0
                    minor_speeds = [] # used to track speed percentiles since they can often vary by a lot

                    while time.time() - tic < NSEC:
                        graph.replay()

                        #g = gn_layer(gn_input)
                        #if not isinstance(gn_layer, GN_NHWC):
                        #    g = act_fn(g)
                        #if BENCH != 'fwd':
                        #    g.sum().backward()

                        torch.cuda.synchronize()
                        ntrials += 1
                        ntrials_minor += 1

                        if time.time() - tic_sec > 0.1:
                            speed = round(ntrials_minor / (time.time() - tic_sec), 2)
                            minor_speeds.append(speed)

                            bw = gn_input.numel() * (3 if BENCH == 'fwd' else 3+5) * {torch.half:2,torch.bfloat16:2, torch.float:4,torch.double:8}[DTYPE]
                            print(f'\t\tBandwidth (GB/s): {ntrials * bw / (time.time() - tic) / 1e9:.2f}, duration: {time.time() - tic:.1f}/{NSEC} seconds completed, speed: {blue(speed)} it/s           \r', end='')
                            #print(f'\t\t{round(time.time() - tic, 1)}/{NSEC} seconds completed, speed: {blue(speed)} it/s\r', end='')
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
                    raise
                    median_speed = slow_speed = fast_speed = '-1 (failed)'
                
                outfile.write(f'{desc},{B},{C},{R},{G},{C//G},{slow_speed},{median_speed},{fast_speed}\n')
            print()
        print(f'All tests done, closing {fname}.')
        outfile.close()


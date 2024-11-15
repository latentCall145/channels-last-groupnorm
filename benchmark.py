# benchmarks various groupnorm kernels
from gnnhwc import GN_NHWC
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, datetime, time, os, itertools

# make strings different colors
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

    dtype_size = torch.finfo(DTYPE).bits / 8
    estimated_mem_usage_gib = (5 * dtype_size * B * C * R * R) / 2**30 #  this is just a rough estimate, likely wrong
    if estimated_mem_usage_gib > 3: # vram filter
        return False
    return True

if __name__ == '__main__':
    INCLUDE_BWD = True # benchmark forward and backward pass
    ACT_FN = 'identity'
    act_fn = {
        'identity': lambda x: x,
        'silu': F.silu,
        'relu': F.relu,
        'gelu': F.gelu,
        'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
    }[ACT_FN]

    NSEC = 1 # number of seconds that each kernel runs for on a certain input
    DTYPES = [torch.float, torch.bfloat16]

    BATCHES = [1, 8]
    CHANNELS = [32, 128]
    RESOLUTIONS = [64, 512]
    NUM_GROUPS = [32]

    GN_KERNELS = [
        (nn.GroupNorm, 'torch.nn GN NCHW'),
        (nn.GroupNorm, 'torch.nn GN NHWC'),
        (GN_NHWC, 'GN NHWC'),
    ]

    os.makedirs('csvs', exist_ok=True)
    fname = datetime.datetime.now().strftime(os.path.join('csvs', '%Y-%m-%d-%H-%M-%S.csv'))
    print(f'Writing to {fname}')
    outfile = open(fname, 'w')
    outfile.write('Kernel,B (batch),C (num channels),R (resolution),G (num groups), D (C/G),Bandwidth (GB/s),Speed (it/s)\n')
    
    configs = list(filter(config_filter, itertools.product(DTYPES, BATCHES, CHANNELS, RESOLUTIONS, NUM_GROUPS)))
    print('Estimated time (seconds) to complete:', NSEC * len(configs) * len(GN_KERNELS))
    print(f'Activation fn: {ACT_FN}')
    print(f'Include bwd pass: {INCLUDE_BWD}')

    for DTYPE, B, C, R, G in configs:
        x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda', requires_grad=True)
        x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).detach().requires_grad_(True)

        print(blue(f'DTYPE: {DTYPE} | B: {B} | C: {C} | R: {R} | G: {G}'))
        for gn_class, desc in GN_KERNELS:
            pad_len = max(len(kernel[1]) for kernel in GN_KERNELS)
            desc = f'{desc:<{pad_len}}' # right pad desc with spaces

            gn_input = x_nchw if 'NCHW' in desc else x_nhwc
            grad = torch.ones_like(gn_input)

            try:
                gn_layer = gn_class(G, C).to(DTYPE).cuda()

                # warmup iter
                g = gn_layer(gn_input)
                if not isinstance(gn_layer, GN_NHWC):
                    g = act_fn(g)
                if INCLUDE_BWD:
                    torch.autograd.grad(g, gn_input, grad_outputs=grad, retain_graph=True)

                tic = time.perf_counter()
                tic_sec = time.perf_counter()
                ntrials = 0

                while time.perf_counter() - tic < NSEC:
                    g = gn_layer(gn_input)
                    if not isinstance(gn_layer, GN_NHWC):
                        g = act_fn(g)
                    if INCLUDE_BWD:
                        torch.autograd.grad(g, gn_input, grad_outputs=grad, retain_graph=True)
                    torch.cuda.synchronize()
                    ntrials += 1

                    if time.perf_counter() - tic_sec > 0.1:
                        speed = round(ntrials / (time.perf_counter() - tic), 2)
                        bw = ntrials * gn_input.nbytes * (3+5 if INCLUDE_BWD else 3) / (time.perf_counter() - tic)
                        bw = round(bw / 1e9, 2)
                        print(f'\t{desc} | {time.perf_counter() - tic:.1f}/{NSEC} sec, bandwidth: {green(bw)} GB/s, speed: {yellow(speed)} it/s           \r', end='')
                        tic_sec = time.perf_counter()

                speed = round(ntrials / (time.perf_counter() - tic), 2)
                bw = ntrials * gn_input.nbytes * (3+5 if INCLUDE_BWD else 3) / (time.perf_counter() - tic)
                bw = round(bw / 1e9, 2)
                print(f'\t{desc} | bandwidth: {green(bw)} GB/s, speed: {yellow(speed)} it/s                             ')
            except KeyboardInterrupt:
                print(f'Keyboard interrupt, closing {fname}                               ')
                outfile.close()
                raise
            except Exception as e:
                print(red(f'\t{desc} | FAILED; err msg:'), str(e).strip())
                raise
                speed = '-1 (failed)'
            
            outfile.write(f'{desc},{B},{C},{R},{G},{C//G},{bw},{speed}\n')
    print(f'All tests done, closing {fname}')
    outfile.close()

# benchmarks various groupnorm kernels
from gnnhwc import GN_NHWC
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, datetime, time, os, itertools, sys
torch.set_printoptions(sci_mode=False, edgeitems=1)

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

    dtype_size = {torch.half: 2, torch.bfloat16: 2, torch.float: 4, torch.double: 8}[DTYPE] # only care about 16/32-bit dtypes for now
    #estimated_mem_usage_gib = (25 * dtype_size * B * C * R * R) / 2**30 #  this is just a rough estimate, likely wrong
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
    DTYPES = [torch.bfloat16]

    BATCHES = [1, 8]
    CHANNELS = [32, 128]
    RESOLUTIONS = [64, 512]
    NUM_GROUPS = [32]

    GN_KERNELS = [
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

        print(blue(f'benchmark (include bwd: {INCLUDE_BWD}) | DTYPE: {DTYPE} | B: {B} | C: {C} | R: {R} | G: {G}'))
        for gn_class, desc in GN_KERNELS:
            gn_input = x_nchw if 'NCHW' in desc else x_nhwc
            grad = torch.ones_like(gn_input)
            print(f'\t{desc}')

            try:
                gn_layer = gn_class(G, C).to(DTYPE).cuda()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    g = gn_layer(gn_input)
                    if not isinstance(gn_layer, GN_NHWC):
                        g = act_fn(g)
                    if INCLUDE_BWD:
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
                    #    g.backward(grad)

                    torch.cuda.synchronize()
                    ntrials += 1
                    ntrials_minor += 1

                    if time.time() - tic_sec > 0.1:
                        speed = round(ntrials_minor / (time.time() - tic_sec), 2)
                        minor_speeds.append(speed)

                        bw1 = gn_input.numel() * (3+5 if INCLUDE_BWD else 3) * {torch.half:2,torch.bfloat16:2, torch.float:4,torch.double:8}[DTYPE]
                        bw = ntrials * gn_input.nbytes * (3+5 if INCLUDE_BWD else 3) / (time.time() - tic)
                        print(f'\t\tduration: {time.time() - tic:.1f}/{NSEC} seconds completed, bandwidth: {blue(round(bw / 1e9, 2))} GB/s, speed: {blue(speed)} it/s           \r', end='')
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

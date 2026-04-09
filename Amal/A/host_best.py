# host_best_optimized.py -- PyOpenCL host for high-performance SGEMM
import numpy as np
import pyopencl as cl
import math, os, random
from time import perf_counter

# ==========================
# Problem sizes
# ==========================
M = 8192
N = 8192
K = 8192

# ==========================
# Kernel tuning parameters
# ==========================
TSM  = 128
TSN  = 128
TSK  = 32   # increased for higher arithmetic intensity
WPTM = 8
WPTN = 8
WIDTH = 4
COUNT = 20

RTSM = TSM // WPTM
RTSN = TSN // WPTN

# ==========================
# Padding utility
# ==========================
def pad_up(x, tile):
    return ((x + tile - 1)//tile)*tile

M_pad = pad_up(M, TSM)
N_pad = pad_up(N, TSN)
K_pad = pad_up(K, TSK)

# ==========================
# Host matrices
# ==========================
AVAL = 3.257
BVAL = 5.723
expected = K*AVAL*BVAL

# Allocate padded arrays
A_pad = np.zeros((M_pad, K_pad), dtype=np.float32)
B_pad = np.zeros((K_pad, N_pad), dtype=np.float32)
A_pad[:M, :K] = AVAL
B_pad[:K, :N] = BVAL

# Pack as K-major float4 for coalesced loads
A_flat = A_pad.T.reshape(K_pad, M_pad//WIDTH, WIDTH).reshape(-1)
B_flat = B_pad.T.reshape(K_pad, N_pad//WIDTH, WIDTH).reshape(-1)

C_np = np.empty((M, N), dtype=np.float32)

# ==========================
# OpenCL setup
# ==========================
ctx = cl.create_some_context(interactive=False)
device = ctx.devices[0]
queue  = cl.CommandQueue(ctx)

print(f"Device : {device.name}, Local mem: {device.local_mem_size//1024} KB")

mf = cl.mem_flags
d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_flat)
d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_flat)
d_C = cl.Buffer(ctx, mf.WRITE_ONLY, C_np.nbytes)

# ==========================
# Load kernel
# ==========================
script_dir  = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(script_dir, "kernel_best.cl")
with open(kernel_path, "r", encoding="utf-8") as f:
    KERNEL_CODE = f.read()

# Inject defines
defines = f"""
#define WIDTH {WIDTH}
#define TSM {TSM}
#define TSN {TSN}
#define TSK {TSK}
#define WPTM {WPTM}
#define WPTN {WPTN}
"""

# Build program with fast math
build_opts = [
    "-cl-fast-relaxed-math",
    "-cl-mad-enable",
    "-cl-no-signed-zeros",
    "-cl-unsafe-math-optimizations"
]

program = cl.Program(ctx, defines + KERNEL_CODE).build(options=build_opts)
kernel = program.gemm_kernel_best
kernel.set_scalar_arg_dtypes([np.int32,np.int32,np.int32,np.int32,np.int32,None,None,None])

# ==========================
# NDRange configuration
# ==========================
local_size  = (RTSM, RTSN)
global_size = ((M_pad//WPTM)//RTSM*RTSM, (N_pad//WPTN)//RTSN*RTSN)  # rounded up to multiples

def run_kernel():
    evt = kernel(queue, global_size, local_size,
                 np.int32(M_pad), np.int32(N_pad), np.int32(K_pad),
                 np.int32(M), np.int32(N),
                 d_A, d_B, d_C)
    return evt

# ==========================
# Warmup & validation
# ==========================
print("Warmup...")
evt = run_kernel()
evt.wait()

cl.enqueue_copy(queue, C_np, d_C).wait()

errors = 0
for _ in range(20):
    i,j = random.randint(0,M-1), random.randint(0,N-1)
    got = float(C_np[i,j])
    ok = abs(got-expected)/abs(expected)<1e-3
    print(f"C[{i},{j}]={got:.4f}, expected={expected:.4f}, {'OK' if ok else 'FAIL'}")
    if not ok:
        errors += 1

if errors==0:
    print("All samples correct")
else:
    print(f"{errors}/20 samples failed")

# ==========================
# Benchmark
# ==========================
print(f"\nBenchmarking {COUNT} iterations...")
t0 = perf_counter()
for _ in range(COUNT):
    evt = run_kernel()
    evt.wait()
elapsed = perf_counter() - t0

gflops = (2.0*M*N*K)/(elapsed/COUNT)/1e9
print(f"Avg time/iter: {elapsed/COUNT*1000:.2f} ms")
print(f"GFLOPS       : {gflops:.1f}")
# host_best.py -- PyOpenCL host for high-performance SGEMM
import numpy as np
import pyopencl as cl
import math, os, random
from time import perf_counter

# ==========================
# Tuning parameters
# ==========================
M = 8192
N = 8192
K = 8192

TSM  = 128
TSN  = 128
TSK  = 16
WPTM = 8
WPTN = 8
WIDTH = 4
COUNT = 30

RTSM = TSM // WPTM
RTSN = TSN // WPTN

# ==========================
# Padding
# ==========================
def pad_up(x, m):
    return math.ceil(x/m)*m

M_pad = pad_up(M, TSM)
N_pad = pad_up(N, TSN)
K_pad = pad_up(K, TSK)

# ==========================
# Test matrices
# ==========================
AVAL = 3.257
BVAL = 5.723
expected = K*AVAL*BVAL

A_np = np.full((M, K), AVAL, dtype=np.float32)
B_np = np.full((K, N), BVAL, dtype=np.float32)

# Pad
A_pad = np.zeros((M_pad, K_pad), dtype=np.float32)
A_pad[:M, :K] = A_np
B_pad = np.zeros((K_pad, N_pad), dtype=np.float32)
B_pad[:K, :N] = B_np

# Pack as float4 (column-major)
A_flat = A_pad.T.reshape(K_pad, M_pad//4, 4).reshape(-1)
B_flat = B_pad.T.reshape(N_pad, K_pad//4, 4).reshape(-1)

C_np = np.empty((M, N), dtype=np.float32)

# ==========================
# OpenCL setup
# ==========================
context = cl.create_some_context(interactive=False)
device  = context.devices[0]
queue   = cl.CommandQueue(context)

print(f"Device : {device.name}, Local mem: {device.local_mem_size//1024} KB")

mf = cl.mem_flags
d_A = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_flat)
d_B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_flat)
d_C = cl.Buffer(context, mf.WRITE_ONLY, C_np.nbytes)

# ==========================
# Load kernel
# ==========================
script_dir  = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(script_dir, "kernel_best.cl")
with open(kernel_path, "r") as f:
    KERNEL_CODE = f.read()

defines = f"""
#define WIDTH {WIDTH}
#define TSM {TSM}
#define TSN {TSN}
#define TSK {TSK}
#define WPTM {WPTM}
#define WPTN {WPTN}
"""

build_opts = ["-cl-fast-relaxed-math", "-cl-mad-enable",
              "-cl-no-signed-zeros", "-cl-unsafe-math-optimizations"]

program = cl.Program(context, defines + KERNEL_CODE).build(options=build_opts)
kernel = program.gemm_kernel_best
kernel.set_scalar_arg_dtypes([np.int32,np.int32,np.int32,np.int32,np.int32,None,None,None])

# ==========================
# Launch configuration
# ==========================
local_size = (RTSM, RTSN)
global_size = ((M_pad//TSM)*RTSM, (N_pad//TSN)*RTSN)

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
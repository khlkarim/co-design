import numpy as np
import pyopencl as cl
from time import time

# ---------------------------
# User-configurable parameters
# ---------------------------
M = 8192
N = 8192
K = 8192

TSM = 128   # Tile size in M
TSN = 128   # Tile size in N
TSK = 16    # Tile size in K
WPTM = 8    # Work per thread M
WPTN = 8    # Work per thread N
WIDTH = 4   # Vectorized load width (float4)

COUNT = 20  # Number of repetitions

# ---------------------------
# Compute padded sizes
# ---------------------------
def pad_to(x, tile):
    return ((x + tile - 1) // tile) * tile

M_XL = pad_to(M, TSM)
N_XL = pad_to(N, TSN)
K_XL = pad_to(K, TSK)

print(f"Padded sizes: M_XL={M_XL}, N_XL={N_XL}, K_XL={K_XL}")

# ---------------------------
# Initialize matrices
# ---------------------------
AVAL = 3.257
BVAL = 5.723
h_A = np.zeros((M_XL, K_XL), dtype=np.float32)
h_B = np.zeros((K_XL, N_XL), dtype=np.float32)
h_C = np.zeros((M_XL, N_XL), dtype=np.float32)

# Fill only original sizes
h_A[:M, :K] = AVAL
h_B[:K, :N] = BVAL

# ---------------------------
# OpenCL setup
# ---------------------------
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_A)
d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_B)
d_C = cl.Buffer(ctx, mf.WRITE_ONLY, h_C.nbytes)

# ---------------------------
# Load Kernel 10 source
# ---------------------------
with open("C:\\co-design\\co-design\\Amal\\A\\kernel_10.cl", "r") as f:
    kernel_source = f.read()

kernel_source = f"""
#define TSM {TSM}
#define TSN {TSN}
#define TSK {TSK}
#define WPTM {WPTM}
#define WPTN {WPTN}
#define WIDTH {WIDTH}
""" + kernel_source

program = cl.Program(ctx, kernel_source).build()
mmul = program.kernel_10  # Use the correct kernel function name from kernel_10.cl

# ---------------------------
# Set kernel argument types (9 arguments)
# ---------------------------
mmul.set_scalar_arg_dtypes([
    np.int32,  # M_XL
    np.int32,  # N_XL
    np.int32,  # K_XL
    np.int32,  # M
    np.int32,  # N
    np.int32,  # K
    None,      # d_A buffer
    None,      # d_B buffer
    None       # d_C buffer
])

# ---------------------------
# Global and local work sizes
# ---------------------------
# Thread-block (local) size
local_size = (TSM // WPTM, TSN // WPTN)

# Global size rounded to full tiles
global_size = (M_XL // WPTM, N_XL // WPTN)

# ---------------------------
# Run benchmark
# ---------------------------
start = time()
for _ in range(COUNT):
    mmul(queue, global_size, local_size,
         M_XL, N_XL, K_XL,  # Padded sizes
         M, N, K,           # Original sizes
         d_A, d_B, d_C)
queue.finish()
elapsed = time() - start

# ---------------------------
# Copy and unpad results
# ---------------------------
cl.enqueue_copy(queue, h_C, d_C)
h_C_final = h_C[:M, :N]

# ---------------------------
# Compute GFLOPS
# ---------------------------
mflop = COUNT * 2.0 * M * N * K / 1e9
gflops = mflop / elapsed

print(f"Time: {elapsed*1000:.2f} ms, Performance: {gflops:.2f} GFLOPS")
print("Sample results (first 5 elements of first row):")
print(h_C_final[0, :5])
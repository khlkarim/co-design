import pyopencl as cl
import numpy as np
import time

# --- Problem sizes ---
M, N, K = 2000, 2000, 2000  # original matrix sizes
TSM, TSN, TSK = 128, 128, 16
WPTM, WPTN = 8, 8
WIDTH = 4  # float4

# --- Helper to round up to multiple ---
def roundup(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

# --- Compute padded sizes ---
M_XL = roundup(M, TSM)
N_XL = roundup(N, TSN)
K_XL = roundup(K, TSK)
K_XL = roundup(K_XL, WIDTH)  # ensure divisible by WIDTH for float4

print(f"Padded sizes: M_XL={M_XL}, N_XL={N_XL}, K_XL={K_XL}")

# --- Random input matrices ---
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

# --- Padding matrices ---
A_pad = np.zeros((M_XL, K_XL), dtype=np.float32)
B_pad = np.zeros((K_XL, N_XL), dtype=np.float32)
A_pad[:M, :K] = A
B_pad[:K, :N] = B
C_pad = np.zeros((M_XL, N_XL), dtype=np.float32)

# --- OpenCL setup ---
platforms = cl.get_platforms()
print("Choose platform:")
for i, p in enumerate(platforms):
    print(f"[{i}] {p}")
plat_idx = int(input("Choice [0]: ") or 0)
platform = platforms[plat_idx]
devices = platform.get_devices()
device = devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

print(f"Using device: {device}")

# --- Load kernel ---
kernel_source = open("kernel_10.cl").read()
prg = cl.Program(ctx, kernel_source).build()

mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_pad)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_pad)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C_pad.nbytes)

# --- Launch parameters ---
local = (TSM // WPTM, TSN // WPTN)

# Make global a multiple of local
global_ = (roundup(M_XL // WPTM, local[0]), roundup(N_XL // WPTN, local[1]))

print(f"Global size: {global_}, Local size: {local}")

# --- Run kernel ---
start = time.time()
prg.mmul(
    queue,               # command queue
    global_,             # global work size
    local,               # local work size
    np.int32(M_XL),      # scalar arguments
    np.int32(N_XL),
    np.int32(K_XL),
    A_buf,               # cl.Buffer objects
    B_buf,
    C_buf
)
queue.finish()
end = time.time()

# --- Copy result back ---
cl.enqueue_copy(queue, C_pad, C_buf)
C = C_pad[:M, :N]  # remove padding

# --- Compute expected sum for verification ---
expected = np.sum(A @ B)

# --- Print random samples ---
print("Random samples:")
for _ in range(10):
    i, j = np.random.randint(0, M), np.random.randint(0, N)
    print(f"C[{i},{j}] = {C[i,j]}")

print(f"\nExpected (approx): {expected:.6e}")
elapsed = end - start
print(f"\nTime: {elapsed:.4f} s")

# --- GFLOPS ---
gflops = 2 * M * N * K / (elapsed * 1e9)
print(f"Performance: {gflops:.2f} GFLOPS")
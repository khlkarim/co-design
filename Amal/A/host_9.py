import numpy as np
import pyopencl as cl
from time import time
import random

# =========================
# 🔧 Parameters (RTX 3050 tuned)
# =========================
M = 2048
N = 2048
K = 2048

TSM = 128
TSN = 128
TSK = 16
WPTM = 8
WPTN = 8
WIDTH = 4

COUNT = 10

# =========================
# ✅ Sanity checks
# =========================
assert M % 4 == 0
assert N % 4 == 0
assert K % 4 == 0

assert M % TSM == 0
assert N % TSN == 0
assert K % TSK == 0

# =========================
# 🧪 Test values
# =========================
AVAL = 3.257
BVAL = 5.723
expected = K * AVAL * BVAL

# =========================
# 🧠 Create matrices
# =========================
A = np.full((K, M), AVAL, dtype=np.float32)
B = np.full((K, N), BVAL, dtype=np.float32)

# =========================
# 🔥 Pack into float4 layout
# =========================
A_vec = A.reshape(K, M // 4, 4).copy()
B_vec = B.reshape(K, N // 4, 4).copy()

# Flatten (IMPORTANT)
A_vec = A_vec.reshape(-1)
B_vec = B_vec.reshape(-1)

# Output
C = np.empty((M, N), dtype=np.float32)

# =========================
# ⚡ OpenCL setup
# =========================
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Buffers
mf = cl.mem_flags
d_A = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_vec)
d_B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_vec)
d_C = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)

# =========================
# 📦 Load kernel
# =========================
with open("C:\\co-design\\co-design\\Amal\\A\\kernel_9.cl") as f:
    kernel_src = f.read()

kernel_src = f"""
#define WIDTH {WIDTH}
#define TSM {TSM}
#define TSN {TSN}
#define TSK {TSK}
#define WPTM {WPTM}
#define WPTN {WPTN}
""" + kernel_src

program = cl.Program(context, kernel_src).build(
    options=["-cl-fast-relaxed-math"]
)

kernel = program.gemm_vec4_prefetch

kernel.set_scalar_arg_dtypes([
    np.int32, np.int32, np.int32,
    None, None, None
])

# =========================
# 🚀 Launch config
# =========================
local_size = (TSM // WPTM, TSN // WPTN)   # (16,16)

global_size = (
    (M // TSM) * local_size[0],
    (N // TSN) * local_size[1]
)

# =========================
# 🏁 Run
# =========================
print("Running kernel...")

start = time()

for _ in range(COUNT):
    kernel(
        queue,
        global_size,
        local_size,
        np.int32(M),
        np.int32(N),
        np.int32(K),
        d_A,
        d_B,
        d_C
    )

queue.finish()

elapsed = time() - start

# =========================
# 📥 Copy result
# =========================
cl.enqueue_copy(queue, C, d_C)

# =========================
# 🧪 Validate
# =========================
print("\nRandom samples:")

for _ in range(10):
    i = random.randint(0, M-1)
    j = random.randint(0, N-1)
    print(f"C[{i},{j}] = {C[i,j]}")

print("\nExpected:", expected)

# =========================
# 📊 Performance
# =========================
flops = COUNT * 2 * M * N * K
gflops = flops / elapsed / 1e9

print(f"\nTime: {elapsed:.4f} s")
print(f"Performance: {gflops:.2f} GFLOPS")
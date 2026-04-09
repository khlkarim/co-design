import numpy
import random
import pyopencl as cl
from time import time


def clamp(x, min_val, max_val):
    return min(max_val, max(min_val, x))


def read_int(prompt, fallback=None):
    try:
        return int(input(prompt))
    except ValueError, TypeError:
        return fallback


# Number of matrix multiplications to perform
COUNT = 1

# Matrix dimensions
DEFAULT_DIM = 8192
M = N = K = DEFAULT_DIM

print("N:", N)

# Number of GFLOP to be performed (multiply + add = factor of 2)
gflop = COUNT * 2.0 * M * N * K / 1e9

# Dummy data: all elements in each matrix are equal
AVAL = 3.257
BVAL = 5.723
CVAL = float(K) * AVAL * BVAL

# --- Device selection ---
use_gpu = read_int("Use GPU? (yes: 1, no: 0, default: 1): ", 1) != 0
use_cpu = read_int("Use CPU? (yes: 1, no: 0, default: 1): ", 1) != 0

if not use_gpu and not use_cpu:
    print("At least one of GPU or CPU must be enabled. Defaulting to GPU only.")
    use_gpu = True

# Partition rows between GPU and CPU
if use_gpu and use_cpu:
    GPU_M = 15 * M // 16
    CPU_M = M // 16
elif use_gpu:
    print("The GPU is going to perform all the work.")
    GPU_M = M
    CPU_M = 0
else:
    print("The CPU is going to perform its normal workload.")
    GPU_M = 0
    CPU_M = M // 16

# Buffer sizes
sizeB = K * N
sizeAGPU = GPU_M * K
sizeACPU = CPU_M * K
sizeCGPU = GPU_M * N
sizeCCPU = CPU_M * N

# --- Kernel configuration ---
gpu_kernel_name = "./B/gpu_kernel.cl"
cpu_kernel_name = "./B/cpu_kernel.cl"

DEFAULT_TS = 16
TS = DEFAULT_TS

if use_gpu:
    print("\nGPU kernel:", gpu_kernel_name)
    TS = read_int(f"TS (4, 8, 16, 32) (default: {DEFAULT_TS}): ", DEFAULT_TS)
    if TS not in [4, 8, 16, 32]:
        print(f"Invalid tile size. Using default: {DEFAULT_TS}")
        TS = DEFAULT_TS
    print(f"Work group size: {TS} x {TS}")

DEFAULT_WIDTH = 4
DEFAULT_TSM = 128
DEFAULT_TSN = 128
DEFAULT_TSK = 32
DEFAULT_WPTM = 8
DEFAULT_WPTN = 8

WIDTH = DEFAULT_WIDTH
TSM = DEFAULT_TSM
TSN = DEFAULT_TSN
TSK = DEFAULT_TSK
WPTM = DEFAULT_WPTM
WPTN = DEFAULT_WPTN

if use_cpu:
    print("\nCPU kernel:", cpu_kernel_name)
    WIDTH = read_int("Work per thread (1, 2, 4) (default: 4): ", DEFAULT_WIDTH)
    if WIDTH not in [1, 2, 4]:
        print(f"Invalid width. Using default: {DEFAULT_WIDTH}")
        WIDTH = DEFAULT_WIDTH
    TSM = read_int(f"TSM  (default: {DEFAULT_TSM}):  ", DEFAULT_TSM)
    TSN = read_int(f"TSN  (default: {DEFAULT_TSN}):  ", DEFAULT_TSN)
    TSK = read_int(f"TSK  (default: {DEFAULT_TSK}):  ", DEFAULT_TSK)
    WPTM = read_int(f"WPTM (default: {DEFAULT_WPTM}): ", DEFAULT_WPTM)
    WPTN = read_int(f"WPTN (default: {DEFAULT_WPTN}): ", DEFAULT_WPTN)

# --- Host buffers ---
gpu_h_B = numpy.full(sizeB, BVAL, dtype=numpy.float32)
cpu_h_B = numpy.full(sizeB, BVAL, dtype=numpy.float32)

gpu_h_A = gpu_h_C = None
cpu_h_A = cpu_h_C = None

if use_gpu:
    gpu_h_A = numpy.full(sizeAGPU, AVAL, dtype=numpy.float32)
    gpu_h_C = numpy.empty(sizeCGPU, dtype=numpy.float32)

if use_cpu:
    cpu_h_A = numpy.full(sizeACPU, AVAL, dtype=numpy.float32)
    cpu_h_C = numpy.empty(sizeCCPU, dtype=numpy.float32)

# --- GPU OpenCL setup ---
gpu_context = gpu_queue = gpu_mmul = None
gpu_a = gpu_b = gpu_c = None

if gpu_h_A is not None and gpu_h_C is not None:
    print("\nCreating a GPU OpenCL context...")
    gpu_kernel_source = f"#define TS {TS}\n" + open(gpu_kernel_name).read()
    gpu_context = cl.create_some_context()
    gpu_queue = cl.CommandQueue(gpu_context)
    gpu_a = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=gpu_h_A)
    gpu_b = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=gpu_h_B)
    gpu_c = cl.Buffer(gpu_context, cl.mem_flags.WRITE_ONLY, size=gpu_h_C.nbytes)
    gpu_program = cl.Program(gpu_context, gpu_kernel_source).build()
    gpu_mmul = gpu_program.mmul
    gpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# --- CPU OpenCL setup ---
cpu_context = cpu_queue = cpu_mmul = None
cpu_a = cpu_b = cpu_c = None

if cpu_h_A is not None and cpu_h_C is not None:
    print("\nCreating a CPU OpenCL context...")
    cpu_kernel_source = f"""
#define WIDTH {WIDTH}
#define TSM {TSM}
#define TSN {TSN}
#define TSK {TSK}
#define WPTM {WPTM}
#define WPTN {WPTN}
""" + open(cpu_kernel_name).read()
    cpu_context = cl.create_some_context()
    cpu_queue = cl.CommandQueue(cpu_context)
    cpu_a = cl.Buffer(cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cpu_h_A)
    cpu_b = cl.Buffer(cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cpu_h_B)
    cpu_c = cl.Buffer(cpu_context, cl.mem_flags.WRITE_ONLY, size=cpu_h_C.nbytes)
    cpu_program = cl.Program(cpu_context, cpu_kernel_source).build()
    cpu_mmul = cpu_program.mmul
    cpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# --- Run multiplications ---
print(f"\nStarting {COUNT} OpenCL Matrix Multiplication(s)...")
start_time = time()

for i in range(COUNT):
    try:
        if gpu_mmul is not None and gpu_queue is not None:
            gpu_mmul(
                gpu_queue,
                (GPU_M, N),
                (TS, TS),
                numpy.int32(GPU_M),
                numpy.int32(N),
                numpy.int32(K),
                gpu_a,
                gpu_b,
                gpu_c,
            )

        if cpu_mmul is not None and cpu_queue is not None:
            cpu_mmul(
                cpu_queue,
                (CPU_M // WPTM, N // WPTN),
                (TSM // WPTM, TSN // WPTN),
                numpy.int32(CPU_M),
                numpy.int32(N),
                numpy.int32(K),
                cpu_a,
                cpu_b,
                cpu_c,
            )

        if gpu_queue is not None:
            gpu_queue.flush()
        if cpu_queue is not None:
            cpu_queue.flush()

        if gpu_queue is not None:
            gpu_queue.finish()
        if cpu_queue is not None:
            cpu_queue.finish()
    except cl.Error as e:
        print(f"OpenCL error: {e}")

run_time = time() - start_time
print(f"End of {COUNT} Matrix Multiplication(s).")
print(f"{run_time:.4f} seconds at {gflop / run_time:.2f} GFLOPS")

# --- Read back results ---
if gpu_queue is not None and gpu_h_C is not None and gpu_c is not None:
    cl.enqueue_copy(gpu_queue, gpu_h_C, gpu_c)

if cpu_queue is not None and cpu_h_C is not None and cpu_c is not None:
    cl.enqueue_copy(cpu_queue, cpu_h_C, cpu_c)

parts = [arr for arr in (gpu_h_C, cpu_h_C) if arr is not None]
h_C = numpy.concatenate(parts)
sizeC = h_C.size

print("\nRandom samples of h_C:")
for _ in range(10):
    idx = random.randint(0, sizeC - 1)
    print(f"\th_C[{idx}] = {h_C[idx]}")

print(f"\tcval: {CVAL}")

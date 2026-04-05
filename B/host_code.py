import numpy
import random
import pyopencl as cl
from time import time


def clamp(x, minVal, maxVal):
    return min(maxVal, max(minVal, x))


def read_int(prompt, fallback=None):
    try:
        n = int(input(prompt))
        return n
    except ValueError, TypeError:
        return fallback


# Number of matrix multiplication to perform
COUNT = 1

# Matrix dimensions
DEFAULT_DIM = 8192

M = DEFAULT_DIM
N = DEFAULT_DIM
K = DEFAULT_DIM

print("N:", N)

# Number of MFLOP to be performed
mflop = COUNT * 2.0 * M * N * K / 1000000.0  # 2.0 because one multiplication and one addition

# Dummy data: All the elements in each matrix are the equal
AVAL = 3.257
BVAL = 5.723
cval = float(K) * AVAL * BVAL

# GPU and CPU workloads
CPU_M = M // 64
GPU_M = 63 * M // 64

sizeAGPU = GPU_M * K
sizeACPU = CPU_M * K
sizeB = K * N
sizeCGPU = GPU_M * N
sizeCCPU = CPU_M * N

# Configuring OpenCL kernel
kernels = [
    "./B/gpu_kernel.cl",
    "./B/cpu_kernel.cl",
]

gpu_kernel_idx = 0
cpu_kernel_idx = 1

gpu_kernel_name = kernels[gpu_kernel_idx]
cpu_kernel_name = kernels[cpu_kernel_idx]

print("GPU kernel:", gpu_kernel_name)
print("CPU kernel:", cpu_kernel_name)

DEFAULT_TS = 16
TS = read_int(f"\nTS (4, 8, 16, 32) (default: {DEFAULT_TS}): ", DEFAULT_TS)

if TS not in [4, 8, 16, 32]:
    TS = DEFAULT_TS
    print("Invalid tile size. Default size", DEFAULT_TS, "will be used.")

print("Work group size is", TS, "*", TS)

gpu_kernel_source = f"""#define TS {TS}\n"""
gpu_kernel_source += open(gpu_kernel_name).read()

# Wider data-types parameters
DEFAULT_WIDTH = 1
WIDTH = read_int("Work per thread (1, 2, 4) (default: 4): ", 4)

if WIDTH not in [1, 2, 4]:
    WIDTH = 4
    print("Invalid width. Default width", WIDTH, "will be used.")

# Register blocking parameters
DEFAULT_TSM = 64
TSM = DEFAULT_TSM  # The tile-size in dimension M
DEFAULT_TSN = 64
TSN = DEFAULT_TSN  # The tile-size in dimension N
DEFAULT_TSK = 32
TSK = DEFAULT_TSK  # The tile-size in dimension K
DEFAULT_WPTM = 8
WPTM = DEFAULT_WPTM  # The work-per-thread in dimension M
DEFAULT_WPTN = 8
WPTN = DEFAULT_WPTN  # The work-per-thread in dimension N

TSM = read_int(f"TSM (default: {DEFAULT_TSM}): ", DEFAULT_TSM)
TSN = read_int(f"TSN (default: {DEFAULT_TSN}): ", DEFAULT_TSN)
TSK = read_int(f"TSK (default: {DEFAULT_TSK}): ", DEFAULT_TSK)
WPTM = read_int(f"WPTM (default: {DEFAULT_WPTM}): ", DEFAULT_WPTM)
WPTN = read_int(f"WPTN (default: {DEFAULT_WPTN}): ", DEFAULT_WPTN)

cpu_kernel_source = f"""
    #define WIDTH {WIDTH}
    #define TSM {TSM}
    #define TSN {TSN}
    #define TSK {TSK}
    #define WPTM {WPTM}
    #define WPTN {WPTN}
    """

cpu_kernel_source += open(cpu_kernel_name).read()

# Create host buffers
gpu_h_A = numpy.empty(sizeAGPU).astype(numpy.float32)
gpu_h_A.fill(AVAL)

cpu_h_A = numpy.empty(sizeACPU).astype(numpy.float32)
cpu_h_A.fill(AVAL)

h_B = numpy.empty(sizeB).astype(numpy.float32)
h_B.fill(BVAL)

gpu_h_C = numpy.empty(sizeCGPU).astype(numpy.float32)
cpu_h_C = numpy.empty(sizeCCPU).astype(numpy.float32)

# Set up OpenCL
print("\nCreating a GPU OpenCL context...")
gpu_context = cl.create_some_context()
gpu_queue = cl.CommandQueue(gpu_context)

# Create GPU buffers
gpu_a = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=gpu_h_A)
gpu_b = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
gpu_c = cl.Buffer(gpu_context, cl.mem_flags.WRITE_ONLY, size=gpu_h_C.nbytes)

gpu_program = cl.Program(gpu_context, gpu_kernel_source).build()

gpu_mmul = gpu_program.mmul
gpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

print("\nCreating a CPU OpenCL context...")
cpu_context = cl.create_some_context()
cpu_queue = cl.CommandQueue(cpu_context)

# Create CPU buffers
cpu_a = cl.Buffer(cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cpu_h_A)
cpu_b = cl.Buffer(cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
cpu_c = cl.Buffer(cpu_context, cl.mem_flags.WRITE_ONLY, size=cpu_h_C.nbytes)

cpu_program = cl.Program(cpu_context, cpu_kernel_source).build()

cpu_mmul = cpu_program.mmul
cpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# Do the multiplication COUNT times
print("\nStarting", COUNT, "OpenCL Matrix Multiplications...")
start_time = time()

for i in range(COUNT):
    try:
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

        gpu_queue.finish()
        cpu_queue.finish()
    except cl.Error as e:
        print(f"OpenCL Error encountered: {e}")

run_time = time() - start_time
print("End of", COUNT, "Matrix Multiplications...")

mflops = mflop / run_time
print(run_time, "seconds at", mflops, "MFLOPS")

# Reading the result h_C
cl.enqueue_copy(gpu_queue, gpu_h_C, gpu_c)
cl.enqueue_copy(cpu_queue, cpu_h_C, cpu_c)

sizeC = sizeCGPU + sizeCCPU
h_C = numpy.concat((gpu_h_C, cpu_h_C))

print("\nRandom samples of h_C:")
numChecks = 10
for i in range(numChecks):
    idx = random.randint(0, sizeC)
    print("\th_C[" + str(idx) + "] = ", h_C[idx])

print("\tcval: ", cval)


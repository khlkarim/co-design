import numpy
import random
import pyopencl as cl
from time import time


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

sizeA = M * K
sizeB = K * N
sizeC = M * N

print("N:", N)

# Number of MFLOP to be performed
mflop = COUNT * 2.0 * M * N * K / 1000000.0  # 2.0 because one multiplication and one addition

# Dummy data: All the elements in each matrix are the equal
AVAL = 3.257
BVAL = 5.723
cval = float(K) * AVAL * BVAL

# GPU and CPU workloads
GPU_BUFFER_SIZE = 63 * sizeC // 64
CPU_BUFFER_SIZE = sizeC // 64

print("GPU_BUFFER_SIZE:", GPU_BUFFER_SIZE)
print("CPU_BUFFER_SIZE:", CPU_BUFFER_SIZE)

# Configuring OpenCL kernel
kernels = [
    "./B/gpu_kernel.cl",
    "./B/optimized_cpu_kernel.cl",
]

gpu_kernel_idx = 0
cpu_kernel_idx = 1

gpu_kernel_name = kernels[gpu_kernel_idx]
cpu_kernel_name = kernels[cpu_kernel_idx]

print("GPU kernel:", gpu_kernel_name)
print("CPU kernel:", cpu_kernel_name)

DEFAULT_TS = 16
TS = DEFAULT_TS

print("Work group size is", TS, "*", TS)

gpu_kernel_source = f"""#define TS {TS}\n"""
gpu_kernel_source += f"""#define GPU_BUFFER_SIZE {GPU_BUFFER_SIZE}\n"""
gpu_kernel_source += open(gpu_kernel_name).read()

cpu_kernel_source = f"""#define TS {TS}\n"""
cpu_kernel_source += f"""#define CPU_BUFFER_SIZE {CPU_BUFFER_SIZE}\n"""
cpu_kernel_source += f"""#define GPU_BUFFER_SIZE {GPU_BUFFER_SIZE}\n"""
cpu_kernel_source += open(cpu_kernel_name).read()

# Create host buffers
h_A = numpy.empty(sizeA).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(sizeB).astype(numpy.float32)
h_B.fill(BVAL)
gpu_h_C = numpy.empty(GPU_BUFFER_SIZE).astype(numpy.float32)
cpu_h_C = numpy.empty(CPU_BUFFER_SIZE).astype(numpy.float32)

# Set up OpenCL
print("\nCreating a GPU OpenCL context...")
gpu_context = cl.create_some_context()
gpu_queue = cl.CommandQueue(gpu_context)

# Create GPU buffers
gpu_a = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
gpu_b = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
gpu_c = cl.Buffer(gpu_context, cl.mem_flags.WRITE_ONLY, size=gpu_h_C.nbytes)

gpu_program = cl.Program(gpu_context, gpu_kernel_source).build()

gpu_mmul = gpu_program.mmul
gpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

print("\nCreating a CPU OpenCL context...")
cpu_context = cl.create_some_context()
cpu_queue = cl.CommandQueue(cpu_context)

# Create CPU buffers
cpu_a = cl.Buffer(cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
cpu_b = cl.Buffer(cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
cpu_c = cl.Buffer(cpu_context, cl.mem_flags.WRITE_ONLY, size=cpu_h_C.nbytes)

cpu_program = cl.Program(cpu_context, cpu_kernel_source).build()

cpu_mmul = cpu_program.mmul
cpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# Do the multiplication COUNT times
print("\nStarting", COUNT, "OpenCL Matrix Multiplications...")
start_time = time()

# Wider data-types parameters
DEFAULT_WIDTH = 4
WIDTH = DEFAULT_WIDTH

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

for i in range(COUNT):
    try:
        gpu_mmul(
            gpu_queue,
            (M, N),
            (TS, TS),
            numpy.int32(M),
            numpy.int32(N),
            numpy.int32(K),
            gpu_a,
            gpu_b,
            gpu_c,
        )
        cpu_mmul(
            cpu_queue,
            (M // WPTM, N // WPTN),
            (TSM // WPTM, TSN // WPTN),
            numpy.int32(M),
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
h_C = numpy.concat((gpu_h_C, cpu_h_C))

print("\nRandom samples of h_C:")
numChecks = 10
for i in range(numChecks):
    idx = random.randint(0, sizeC)
    print("\th_C[" + str(idx) + "] = ", h_C[idx])

print("\tcval: ", cval)


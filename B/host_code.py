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

# Number of MFLOP to be performed
mflop = COUNT * 2.0 * M * N * K / 1000000.0  # 2.0 because one multiplication and one addition

# Dummy data: All the elements in each matrix are the equal
AVAL = 3.257
BVAL = 5.723
cval = float(K) * AVAL * BVAL

# Configuring OpenCL kernel
kernels = [
    "./B/default_uncoalsced_kernel.cl",  # Default uncoalsced kernel
]

print("Available kernels:")
print("\t 0: Default uncoalsced kernel")

kernel_source = ""
kernel_name = kernels[0]
kernelIdx = read_int("Pick a kernel (default: 0): ", 0)

if kernelIdx not in [0]:
    kernelIdx = 0
    print("Invalid kernel idx:", kernelIdx)
    print("Falling back to the default coalsced kernel.")

DEFAULT_TS = 16
TS = read_int(f"\nTS (4, 8, 16, 32) (default: {DEFAULT_TS}): ", DEFAULT_TS)

if TS not in [4, 8, 16, 32]:
    TS = DEFAULT_TS
    print("Invalid tile size. Default size", DEFAULT_TS, "will be used.")

print("Block Size is", TS, "*", TS)
kernel_source += f"""#define TS {TS}\n"""
kernel_source += open(kernels[kernelIdx]).read()

# Set up OpenCL
print("\nCreating an OpenCL context...")
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Create host buffers
h_A = numpy.empty(sizeA).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(sizeB).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(sizeC).astype(numpy.float32)

# Create device buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

program = cl.Program(context, kernel_source).build()

mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# Do the multiplication COUNT times
print("\nStarting", COUNT, "OpenCL Matrix Multiplications...")

start_time = time()
for i in range(COUNT):
    try:
        mmul(
            queue,
            (M, N),
            (TS, TS),
            numpy.int32(M),
            numpy.int32(N),
            numpy.int32(K),
            d_a,
            d_b,
            d_c,
        )
        queue.finish()
    except cl.Error as e:
        print(f"OpenCL Error encountered: {e}")
run_time = time() - start_time

print("End of", COUNT, "Matrix Multiplications...")

mflops = mflop / run_time
print(run_time, "seconds at", mflops, "MFLOPS")

# Reading the result h_C
cl.enqueue_copy(queue, h_C, d_c)

print("\nRandom samples of h_C:")
numChecks = 10
for i in range(numChecks):
    idx = random.randint(0, sizeC)
    print("\th_C[" + str(idx) + "] = ", h_C[idx])

print("\tcval: ", cval)


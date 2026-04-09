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

M = read_int(f"M (default: {DEFAULT_DIM}): ", DEFAULT_DIM)
N = read_int(f"N (default: {DEFAULT_DIM}): ", DEFAULT_DIM)
K = read_int(f"K (default: {DEFAULT_DIM}): ", DEFAULT_DIM)

sizeA = M * K
sizeB = K * N
sizeC = M * N

# Number of MFLOP to be performed
gflop = COUNT * 2.0 * M * N * K / 1e9  # 2.0 because one multiplication and one addition

# Dummy data: All the elements in each matrix are the equal
AVAL = 3.257
BVAL = 5.723
cval = float(K) * AVAL * BVAL

# Configuring OpenCL kernel
kernels = [
    "./A/kernel_1.cl",  # Default coalsced kernel
    "./A/kernel_4.cl",  # Wider data-types
    "./A/kernel_6.cl",  # 2D register blocking
    "./A/kernel_7.cl",  # Wider loads with register blocking
]

# The block size for kernels 1 and 4
DEFAULT_TS = 16
TS = DEFAULT_TS

# Wider data-types parameters
DEFAULT_WIDTH = 1
WIDTH = DEFAULT_WIDTH

# Register blocking parameters
DEFAULT_TSM = 128
TSM = DEFAULT_TSM  # The tile-size in dimension M
DEFAULT_TSN = 128
TSN = DEFAULT_TSN  # The tile-size in dimension N
DEFAULT_TSK = 32
TSK = DEFAULT_TSK  # The tile-size in dimension K
DEFAULT_WPTM = 8
WPTM = DEFAULT_WPTM  # The work-per-thread in dimension M
DEFAULT_WPTN = 8
WPTN = DEFAULT_WPTN  # The work-per-thread in dimension N

print("\nAvailable kernels:")
print("\t 0: Default coalsced kernel")
print("\t 1: Wider data-types kernel")
print("\t 2: Register blocking kernel")
print("\t 3: Wider loads with register blocking kernel")

kernel_source = ""
kernel_name = kernels[0]
kernelIdx = read_int("Pick a kernel (default: 0): ", 0)

if kernelIdx not in [0, 1, 2, 3]:
    kernelIdx = 0
    print("Invalid kernel idx:", kernelIdx)
    print("Falling back to the default coalsced kernel.")

if kernelIdx == 0 or kernelIdx == 1:
    TS = read_int(f"\nTS (4, 8, 16, 32) (default: {DEFAULT_TS}): ", DEFAULT_TS)

    if TS not in [4, 8, 16, 32]:
        TS = DEFAULT_TS
        print("Invalid tile size. Default size", DEFAULT_TS, "will be used.")

    print("Block Size is", TS, "*", TS)
    kernel_source += f"""#define TS {TS}\n"""

if kernelIdx == 1 or kernelIdx == 3:
    WIDTH = read_int("Work per thread (1, 2, 4) (default: 4): ", 4)
    if WIDTH not in [1, 2, 4]:
        WIDTH = 4
        print("Invalid width. Default width", WIDTH, "will be used.")

    kernel_source += f"""#define WIDTH {WIDTH}\n"""

if kernelIdx == 2 or kernelIdx == 3:
    TSM = read_int(f"TSM (default: {DEFAULT_TSM}): ", DEFAULT_TSM)
    TSN = read_int(f"TSN (default: {DEFAULT_TSN}): ", DEFAULT_TSN)
    TSK = read_int(f"TSK (default: {DEFAULT_TSK}): ", DEFAULT_TSK)
    WPTM = read_int(f"WPTM (default: {DEFAULT_WPTM}): ", DEFAULT_WPTM)
    WPTN = read_int(f"WPTN (default: {DEFAULT_WPTN}): ", DEFAULT_WPTN)

    kernel_source += f"""
        #define TSM {TSM}
        #define TSN {TSN}
        #define TSK {TSK}
        #define WPTM {WPTM}
        #define WPTN {WPTN}
        """

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
        if kernelIdx == 0:
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
        elif kernelIdx == 1:
            mmul(
                queue,
                (M // WIDTH, N),
                (TS // WIDTH, TS),
                numpy.int32(M),
                numpy.int32(N),
                numpy.int32(K),
                d_a,
                d_b,
                d_c,
            )
        elif kernelIdx == 2 or kernelIdx == 3:
            mmul(
                queue,
                (M // WPTM, N // WPTN),
                (TSM // WPTM, TSN // WPTN),
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

gflops = gflop / run_time
print(run_time, "seconds at", gflops, "GFLOPS")

# Reading the result h_C
cl.enqueue_copy(queue, h_C, d_c)

print("\nRandom samples of h_C:")
numChecks = 10
for i in range(numChecks):
    idx = random.randint(0, sizeC)
    print("\th_C[" + str(idx) + "] = ", h_C[idx])

print("\tcval: ", cval)

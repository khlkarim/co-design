import numpy
import pyopencl as cl
from time import time


def read_int(prompt, fallback=None):
    try:
        n = int(input(prompt))
        return n
    except ValueError, TypeError:
        return fallback


# Number of matrix multiplication to perform
COUNT = 20

# Matrix dimensions
DEFAULT_DIM = 2048

M = read_int(f"M (default: {DEFAULT_DIM}): ", DEFAULT_DIM)
N = read_int(f"N (default: {DEFAULT_DIM}): ", DEFAULT_DIM)
K = read_int(f"K (default: {DEFAULT_DIM}): ", DEFAULT_DIM)

sizeA = M * K
sizeB = K * N
sizeC = M * N

# Number of MFLOP to be performed
mflop = COUNT * 2.0 * M * N * K / 1000000.0  # 2.0 because one multiplication and one addition

# Used in the wider data-types kernel
DEFAULT_WIDTH = 1
width = DEFAULT_WIDTH

# Dummy data: All the elements in each matrix are the same
AVAL = 3.257
BVAL = 5.723
cval = float(K) * AVAL * BVAL

# localsize ** 2 = work group size?
DEFAULT_LOCALSIZE = 16
localsize = read_int(f"Localsize (4, 8, 16, 32) (default: {DEFAULT_LOCALSIZE}): ", DEFAULT_LOCALSIZE)

if localsize not in [4, 8, 16, 32]:
    localsize = DEFAULT_LOCALSIZE
    print("Invalid localsize size. Default Size", DEFAULT_LOCALSIZE, "will be used.")

print("Block Size is", localsize, "*", localsize)

# OpenCL kernel
kernel_source = ""
kernel_name = "./matrix_multiplication_kernel_optimization/default_coalsced_kernel.cl"

print("Available kernels:")
print("\t 0: default coalsced kernel")
print("\t 1: optimized coalsced kernel 1 - Tiled")
print("\t 2: optimized coalsced kernel 2 - Use wider data types")

kernelIdx = read_int("Pick a kernel (default: 0): ", 0)

if kernelIdx == 1:
    kernel_source += "#define TS " + str(localsize) + "\n"
    kernel_name = "./matrix_multiplication_kernel_optimization/optimized_coalsced_kernel_1.cl"

elif kernelIdx == 2:
    width = read_int("Work per thread (1, 2, 4): ", 4)
    if width not in [1, 2, 4]:
        width = DEFAULT_WIDTH
        print("Invalid width. Default width", DEFAULT_WIDTH, "will be used.")

    kernel_source += "#define TS " + str(localsize) + "\n#define WIDTH " + str(width) + "\n"
    kernel_name = "./matrix_multiplication_kernel_optimization/optimized_coalsced_kernel_2.cl"

elif kernelIdx != 0:
    print("Invalid kernel idx:", kernelIdx)
    print("Falling back to the default coalsced kernel.")

kernel_source += open(kernel_name).read()

# Set up OpenCL
print("Creating an OpenCL context...")
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
print("Starting", COUNT, "OpenCL Matrix Multiplications...")

start_time = time()
for i in range(COUNT):
    try:
        mmul(
            queue,
            (M // width, N),
            (localsize // width, localsize),
            numpy.int32(M),
            numpy.int32(N),
            numpy.int32(K),
            d_a,
            d_b,
            d_c,
        )
        queue.finish()
    except Exception:
        print(" ===  Error for localsize =", localsize, "===\n")
run_time = time() - start_time

print("End of", COUNT, "Matrix Multiplications...")

mflops = mflop / run_time
print(run_time, "seconds at", mflops, "MFLOPS")

# Reading the result h_C
cl.enqueue_copy(queue, h_C, d_c)

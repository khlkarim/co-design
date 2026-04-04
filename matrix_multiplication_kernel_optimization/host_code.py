import numpy
import pyopencl as cl
from time import time

# Number of matrix multiplication to perform
COUNT = 20

# Matrix dimension
N = 2048
size = N * N

# Number of MFLOP to be performed
mflop = COUNT * 2.0 * N * N * N / 1000000.0  # why 2.0? isn't it a fused multiply add?

# Dummy data: All the elements in each matrix are the same
AVAL = 3.257
BVAL = 5.723

# OpenCL kernel
kernel_name = "./matrix_multiplication_kernel_optimization/default_coalsced_kernel.cl"

# localsize ** 2 = work group size?
kernel_size = input(
    "Please enter a value for localsize. Possible values: 4, 8, 16 and 32 : "
)

localsize = 16
if kernel_size in ["4", "8", "16", "32"]:
    localsize = int(kernel_size)
    print("Blocks Size is", localsize, "*", localsize)
else:
    print("=== No valid input. Default Size 16 will be used. Block Size = 16*16")


# Set up OpenCL
print("Creating an OpenCL context...")
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Create host buffers
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

# Create device buffers
d_a = cl.Buffer(
    context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A
)
d_b = cl.Buffer(
    context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B
)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

kernelsource = open(kernel_name).read()
program = cl.Program(context, kernelsource).build()

mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])

# Do the multiplication COUNT times
print("Starting", COUNT, "OpenCL Matrix Multiplications...")

start_time = time()
for i in range(COUNT):
    try:
        mmul(queue, (N, N), (localsize, localsize), numpy.int32(N), d_a, d_b, d_c)
        queue.finish()
    except Exception:
        print(" ===  Error for localsize =", localsize, "===\n")
run_time = time() - start_time

print("End of", COUNT, "Matrix Multiplications...")

mflops = mflop / run_time
print(run_time, "seconds at", mflops, "MFLOPS")

# Reading the result h_C
cl.enqueue_copy(queue, h_C, d_c)

import numpy
import pyopencl as cl
from time import time

# Number of matrix multiplication to perform
COUNT = 1

# Matrix dimension
N = 8192
M = N
K = N
sizeA = M * K
sizeB = K * N
sizeC = M * N

# Number of MFLOP to be performed
mflop = COUNT * 2.0 * M * N * K / 1000000.0  # why 2.0? isn't it a fused multiply add?

# Dummy data: All the elements in each matrix are the same
AVAL = 3.257
BVAL = 5.723
cval = float(K) * AVAL * BVAL

# OpenCL kernel
kernel_name = "./running_on_multiple_opencl_devices/default_uncoalsced_kernel.cl"

# localsize ** 2 = work group size?
localsize = 16

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

kernelsource = open(kernel_name).read()
program = cl.Program(context, kernelsource).build()

mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# Do the multiplication COUNT times
print("Starting", COUNT, "OpenCL Matrix Multiplications...")

start_time = time()
for i in range(COUNT):
    try:
        mmul(
            queue,
            (M, N),
            (localsize, localsize),
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

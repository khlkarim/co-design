// kernel_10.cl
#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#endif

#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#define LPTB ((TSK*TSN)/(RTSM*RTSN))

// -----------------------------------------------------------
// Matrix multiplication with prefetching and 2D register blocking
// -----------------------------------------------------------
__kernel void mmul(
    const int M_XL,
    const int N_XL,
    const int K_XL,
    const __global floatX* A,
    const __global floatX* B,
    __global float* C
) {
    // Thread identifiers
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int offsetM = TSM*get_group_id(0);
    const int offsetN = TSN*get_group_id(1);

    // Double-buffered local memory for prefetch
    __local float Asub[2][TSK*TSM];
    __local float Bsub[2][TSK*TSN];

    // Registers
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Init accumulators
    for (int wm=0; wm<WPTM; wm++)
        for (int wn=0; wn<WPTN; wn++)
            acc[wm][wn] = 0.0f;

    int numTiles = K_XL / TSK;

    // Load first tile (tile 0)
    for (int la=0; la<LPTA/WIDTH; la++) {
        int tid = tidn*RTSM + tidm;
        int id = la*RTSN*RTSM + tid;
        int row = id % (TSM/WIDTH);
        int col = id / (TSM/WIDTH);
        int tiledIndex = col + 0*TSK;

        floatX vecA = A[tiledIndex*(M_XL/WIDTH) + offsetM/WIDTH + row];
        floatX vecB = B[tiledIndex*(N_XL/WIDTH) + offsetN/WIDTH + row];

#if WIDTH == 1
        Asub[0][col*TSM + row] = vecA;
        Bsub[0][col*TSN + row] = vecB;
#elif WIDTH == 2
        Asub[0][col*TSM + 2*row + 0] = vecA.x;
        Asub[0][col*TSM + 2*row + 1] = vecA.y;
        Bsub[0][col*TSN + 2*row + 0] = vecB.x;
        Bsub[0][col*TSN + 2*row + 1] = vecB.y;
#elif WIDTH == 4
        Asub[0][col*TSM + 4*row + 0] = vecA.x;
        Asub[0][col*TSM + 4*row + 1] = vecA.y;
        Asub[0][col*TSM + 4*row + 2] = vecA.z;
        Asub[0][col*TSM + 4*row + 3] = vecA.w;

        Bsub[0][col*TSN + 4*row + 0] = vecB.x;
        Bsub[0][col*TSN + 4*row + 1] = vecB.y;
        Bsub[0][col*TSN + 4*row + 2] = vecB.z;
        Bsub[0][col*TSN + 4*row + 3] = vecB.w;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t=0; t<numTiles; t++) {
        int tile_idx = t % 2;
        int next_tile = (t+1) % 2;

        // Prefetch next tile
        if (t+1 < numTiles) {
            for (int la=0; la<LPTA/WIDTH; la++) {
                int tid = tidn*RTSM + tidm;
                int id = la*RTSN*RTSM + tid;
                int row = id % (TSM/WIDTH);
                int col = id / (TSM/WIDTH);
                int tiledIndex = col + (t+1)*TSK;

                floatX vecA = A[tiledIndex*(M_XL/WIDTH) + offsetM/WIDTH + row];
                floatX vecB = B[tiledIndex*(N_XL/WIDTH) + offsetN/WIDTH + row];

#if WIDTH == 1
                Asub[next_tile][col*TSM + row] = vecA;
                Bsub[next_tile][col*TSN + row] = vecB;
#elif WIDTH == 2
                Asub[next_tile][col*TSM + 2*row + 0] = vecA.x;
                Asub[next_tile][col*TSM + 2*row + 1] = vecA.y;
                Bsub[next_tile][col*TSN + 2*row + 0] = vecB.x;
                Bsub[next_tile][col*TSN + 2*row + 1] = vecB.y;
#elif WIDTH == 4
                Asub[next_tile][col*TSM + 4*row + 0] = vecA.x;
                Asub[next_tile][col*TSM + 4*row + 1] = vecA.y;
                Asub[next_tile][col*TSM + 4*row + 2] = vecA.z;
                Asub[next_tile][col*TSM + 4*row + 3] = vecA.w;

                Bsub[next_tile][col*TSN + 4*row + 0] = vecB.x;
                Bsub[next_tile][col*TSN + 4*row + 1] = vecB.y;
                Bsub[next_tile][col*TSN + 4*row + 2] = vecB.z;
                Bsub[next_tile][col*TSN + 4*row + 3] = vecB.w;
#endif
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute current tile
        for (int k=0; k<TSK; k++) {
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[tile_idx][k*TSN + col];
            }
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[tile_idx][k*TSM + row];
                for (int wn=0; wn<WPTN; wn++)
                    acc[wm][wn] += Areg * Breg[wn];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store result, respecting padded size
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            if (globalRow < M_XL && globalCol < N_XL)
                C[globalCol*M_XL + globalRow] = acc[wm][wn];
        }
    }
}
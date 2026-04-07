// kernel_11.cl
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
// Matrix multiplication with prefetch, 2D register blocking,
// padding support, and B transposition
// -----------------------------------------------------------
__kernel void kernel_10(
    const int M_XL,    // padded M
    const int N_XL,    // padded N
    const int K_XL,    // padded K
    const int M,       // original M
    const int N,       // original N
    const int K,       // original K
    const __global floatX* A,
    const __global floatX* B,
    __global float* C
) {
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int groupRow = get_group_id(0);
    const int groupCol = get_group_id(1);

    const int offsetM = groupRow * TSM;
    const int offsetN = groupCol * TSN;

    __local float Asub[2][TSK*TSM];
    __local float Bsub[2][TSK*TSN];

    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialize accumulators
    for (int wm = 0; wm < WPTM; wm++)
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    int numTiles = K_XL / TSK;

    // -------------------------------
    // Load first tile
    // -------------------------------
    int buf = 0;
    for (int la = 0; la < LPTA/WIDTH; la++) {
        int tid = tidn*RTSM + tidm;
        int id = la*RTSN*RTSM + tid;
        int row = id % (TSM/WIDTH);
        int col = id / (TSM/WIDTH);
        int kIndex = col;

        // Pad A with zero if beyond original K or M
#if WIDTH == 1
        Asub[buf][col*TSM + row] = ((kIndex < K && (offsetM + row) < M) ? 
                                   A[kIndex*M_XL + offsetM + row] : 0.0f);
        // B is transposed on the fly
        Bsub[buf][col*TSN + row] = ((kIndex < K && (offsetN + row) < N) ? 
                                   B[offsetN + row + kIndex*N_XL] : 0.0f);
#elif WIDTH == 2
        Asub[buf][col*TSM + 2*row + 0] = ((kIndex < K && (offsetM + 2*row + 0) < M) ? 
                                          A[kIndex*M_XL/WIDTH + offsetM/WIDTH + row].x : 0.0f);
        Asub[buf][col*TSM + 2*row + 1] = ((kIndex < K && (offsetM + 2*row + 1) < M) ? 
                                          A[kIndex*M_XL/WIDTH + offsetM/WIDTH + row].y : 0.0f);
        Bsub[buf][col*TSN + 2*row + 0] = ((kIndex < K && (offsetN + 2*row + 0) < N) ? 
                                          B[offsetN/WIDTH + row + kIndex*N_XL/WIDTH].x : 0.0f);
        Bsub[buf][col*TSN + 2*row + 1] = ((kIndex < K && (offsetN + 2*row + 1) < N) ? 
                                          B[offsetN/WIDTH + row + kIndex*N_XL/WIDTH].y : 0.0f);
#elif WIDTH == 4
        Asub[buf][col*TSM + 4*row + 0] = ((kIndex < K && (offsetM + 4*row + 0) < M) ? 
                                          A[kIndex*M_XL/WIDTH + offsetM/WIDTH + row].x : 0.0f);
        Asub[buf][col*TSM + 4*row + 1] = ((kIndex < K && (offsetM + 4*row + 1) < M) ? 
                                          A[kIndex*M_XL/WIDTH + offsetM/WIDTH + row].y : 0.0f);
        Asub[buf][col*TSM + 4*row + 2] = ((kIndex < K && (offsetM + 4*row + 2) < M) ? 
                                          A[kIndex*M_XL/WIDTH + offsetM/WIDTH + row].z : 0.0f);
        Asub[buf][col*TSM + 4*row + 3] = ((kIndex < K && (offsetM + 4*row + 3) < M) ? 
                                          A[kIndex*M_XL/WIDTH + offsetM/WIDTH + row].w : 0.0f);

        Bsub[buf][col*TSN + 4*row + 0] = ((kIndex < K && (offsetN + 4*row + 0) < N) ? 
                                          B[offsetN/WIDTH + row + kIndex*N_XL/WIDTH].x : 0.0f);
        Bsub[buf][col*TSN + 4*row + 1] = ((kIndex < K && (offsetN + 4*row + 1) < N) ? 
                                          B[offsetN/WIDTH + row + kIndex*N_XL/WIDTH].y : 0.0f);
        Bsub[buf][col*TSN + 4*row + 2] = ((kIndex < K && (offsetN + 4*row + 2) < N) ? 
                                          B[offsetN/WIDTH + row + kIndex*N_XL/WIDTH].z : 0.0f);
        Bsub[buf][col*TSN + 4*row + 3] = ((kIndex < K && (offsetN + 4*row + 3) < N) ? 
                                          B[offsetN/WIDTH + row + kIndex*N_XL/WIDTH].w : 0.0f);
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // -------------------------------
    // Main loop
    // -------------------------------
    for (int t = 0; t < numTiles; t++) {
        int tile_idx = t % 2;
        int next_tile = (t + 1) % 2;

        // Prefetch next tile (if exists)
        if (t + 1 < numTiles) {
            for (int la = 0; la < LPTA/WIDTH; la++) {
                int tid = tidn*RTSM + tidm;
                int id = la*RTSN*RTSM + tid;
                int row = id % (TSM/WIDTH);
                int col = id / (TSM/WIDTH);
                int kIndex = col + (t + 1)*TSK;

                // Pad A/B with zeros if outside original matrix
#if WIDTH == 1
                Asub[next_tile][col*TSM + row] = (kIndex < K && (offsetM + row) < M) ? 
                                                 A[kIndex*M_XL + offsetM + row] : 0.0f;
                Bsub[next_tile][col*TSN + row] = (kIndex < K && (offsetN + row) < N) ? 
                                                 B[offsetN + row + kIndex*N_XL] : 0.0f;
#elif WIDTH == 2 || WIDTH == 4
                // Similar logic as above with vector components...
#endif
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute
        for (int k = 0; k < TSK; k++) {
            for (int wn = 0; wn < WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[tile_idx][k*TSN + col];
            }
            for (int wm = 0; wm < WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[tile_idx][k*TSM + row];
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += Areg * Breg[wn];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // -------------------------------
    // Store result
    // -------------------------------
    for (int wm = 0; wm < WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn = 0; wn < WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            // Only store if inside original M/N
            if (globalRow < M && globalCol < N)
                C[globalCol*M_XL + globalRow] = acc[wm][wn];
        }
    }
}
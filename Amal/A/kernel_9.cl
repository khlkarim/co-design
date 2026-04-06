#define WIDTH 4

#define TSM 128
#define TSN 128
#define TSK 16

#define WPTM 8
#define WPTN 8

#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)

#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#define LPTB ((TSK*TSN)/(RTSM*RTSN))

#define floatX float4

#define A_IDX(k,m) ((k)*TSM + (m))
#define B_IDX(k,n) ((k)*TSN + (n))

__kernel void gemm_vec4_prefetch(
    const int M, const int N, const int K,
    const __global floatX* A,
    const __global floatX* B,
    __global float* C)
{
    // Thread IDs
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);

    const int groupRow = get_group_id(0);
    const int groupCol = get_group_id(1);

    // Local memory (double buffered)
    __local float Asub[2][TSK*TSM];
    __local float Bsub[2][TSK*TSN];

    // Registers
    float acc[WPTM][WPTN] = {0};
    float Areg;
    float Breg[WPTN];

    const int offsetM = groupRow * TSM;
    const int offsetN = groupCol * TSN;

    const int numTiles = K / TSK;

    // ============================
    // 🔹 Load FIRST tile
    // ============================
    int buf = 0;

    for (int la = 0; la < LPTA/WIDTH; la++) {
        int tid = tidn*RTSM + tidm;
        int id = la*RTSN*RTSM + tid;

        int row = id % (TSM/WIDTH);
        int col = id / (TSM/WIDTH);

        int tiledIndex = col;

        float4 vecA = A[(tiledIndex)*(M/WIDTH) + (offsetM/WIDTH) + row];
        float4 vecB = B[(tiledIndex)*(N/WIDTH) + (offsetN/WIDTH) + row];

        // Store A
        Asub[buf][A_IDX(col, WIDTH*row+0)] = vecA.x;
        Asub[buf][A_IDX(col, WIDTH*row+1)] = vecA.y;
        Asub[buf][A_IDX(col, WIDTH*row+2)] = vecA.z;
        Asub[buf][A_IDX(col, WIDTH*row+3)] = vecA.w;

        // Store B
        Bsub[buf][B_IDX(col, WIDTH*row+0)] = vecB.x;
        Bsub[buf][B_IDX(col, WIDTH*row+1)] = vecB.y;
        Bsub[buf][B_IDX(col, WIDTH*row+2)] = vecB.z;
        Bsub[buf][B_IDX(col, WIDTH*row+3)] = vecB.w;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // ============================
    // 🔹 Main loop
    // ============================
    for (int t = 0; t < numTiles; t++) {

        int buf = t % 2;
        int nextBuf = (t + 1) % 2;

        // 🔸 Prefetch next tile
        if (t + 1 < numTiles) {
            for (int la = 0; la < LPTA/WIDTH; la++) {

                int tid = tidn*RTSM + tidm;
                int id = la*RTSN*RTSM + tid;

                int row = id % (TSM/WIDTH);
                int col = id / (TSM/WIDTH);

                int tiledIndex = TSK*(t+1) + col;

                float4 vecA = A[(tiledIndex)*(M/WIDTH) + (offsetM/WIDTH) + row];
                float4 vecB = B[(tiledIndex)*(N/WIDTH) + (offsetN/WIDTH) + row];

                Asub[nextBuf][A_IDX(col, WIDTH*row+0)] = vecA.x;
                Asub[nextBuf][A_IDX(col, WIDTH*row+1)] = vecA.y;
                Asub[nextBuf][A_IDX(col, WIDTH*row+2)] = vecA.z;
                Asub[nextBuf][A_IDX(col, WIDTH*row+3)] = vecA.w;

                Bsub[nextBuf][B_IDX(col, WIDTH*row+0)] = vecB.x;
                Bsub[nextBuf][B_IDX(col, WIDTH*row+1)] = vecB.y;
                Bsub[nextBuf][B_IDX(col, WIDTH*row+2)] = vecB.z;
                Bsub[nextBuf][B_IDX(col, WIDTH*row+3)] = vecB.w;
            }
        }

        // 🔸 Compute
        for (int k = 0; k < TSK; k++) {

            // Load B into registers
            for (int wn = 0; wn < WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[buf][B_IDX(k, col)];
            }

            // Multiply
            for (int wm = 0; wm < WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[buf][A_IDX(k, row)];

                for (int wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ============================
    // 🔹 Store results
    // ============================
    for (int wm = 0; wm < WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;

        for (int wn = 0; wn < WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;

            C[globalRow * N + globalCol] = acc[wm][wn];
        }
    }
}
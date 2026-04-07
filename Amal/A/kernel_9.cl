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

// 🔥 add padding to kill bank conflicts
#define PADM (TSM + 1)
#define PADN (TSN + 1)

#define A_IDX(k,m) ((k)*PADM + (m))
#define B_IDX(k,n) ((k)*PADN + (n))

__kernel void gemm_vec4_prefetch(
    const int M, const int N, const int K,
    const __global floatX* A,
    const __global floatX* B,
    __global float* C)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);

    const int groupRow = get_group_id(0);
    const int groupCol = get_group_id(1);

    __local float Asub[2][TSK * PADM];
    __local float Bsub[2][TSK * PADN];

    float acc[WPTM][WPTN] = {0};
    float Areg;
    float Breg[WPTN];

    const int offsetM = groupRow * TSM;
    const int offsetN = groupCol * TSN;

    const int numTiles = K / TSK;

    int buf = 0;

    // =====================================
    // 🔹 Load FIRST tile
    // =====================================
    for (int la = 0; la < LPTA / WIDTH; la++) {

        int tid = tidn * RTSM + tidm;
        int id  = la * RTSN * RTSM + tid;

        // ✅ A indexing (correct)
        int rowA = id % (TSM / WIDTH);
        int colA = id / (TSM / WIDTH);

        float4 vecA = A[(colA) * (M / WIDTH) + (offsetM / WIDTH) + rowA];

        vstore4(vecA, 0, &Asub[buf][A_IDX(colA, rowA * WIDTH)]);

        // 🔥 FIX: B must use TSN (NOT TSM)
        int rowB = id % (TSN / WIDTH);
        int colB = id / (TSN / WIDTH);

        float4 vecB = B[(colB) * (N / WIDTH) + (offsetN / WIDTH) + rowB];

        vstore4(vecB, 0, &Bsub[buf][B_IDX(colB, rowB * WIDTH)]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // =====================================
    // 🔹 Main loop
    // =====================================
    for (int t = 0; t < numTiles; t++) {

        buf = t % 2;                  // 🔥 FIX: no shadowing
        int nextBuf = (t + 1) % 2;

        // 🔸 Prefetch next tile
        if (t + 1 < numTiles) {

            for (int la = 0; la < LPTA / WIDTH; la++) {

                int tid = tidn * RTSM + tidm;
                int id  = la * RTSN * RTSM + tid;

                // A
                int rowA = id % (TSM / WIDTH);
                int colA = id / (TSM / WIDTH);

                int tiledIndexA = TSK * (t + 1) + colA;

                float4 vecA = A[(tiledIndexA) * (M / WIDTH) + (offsetM / WIDTH) + rowA];

                vstore4(vecA, 0, &Asub[nextBuf][A_IDX(colA, rowA * WIDTH)]);

                // 🔥 FIXED B
                int rowB = id % (TSN / WIDTH);
                int colB = id / (TSN / WIDTH);

                int tiledIndexB = TSK * (t + 1) + colB;

                float4 vecB = B[(tiledIndexB) * (N / WIDTH) + (offsetN / WIDTH) + rowB];

                vstore4(vecB, 0, &Bsub[nextBuf][B_IDX(colB, rowB * WIDTH)]);
            }
        }

        // 🔸 Compute
        for (int k = 0; k < TSK; k++) {

            for (int wn = 0; wn < WPTN; wn++) {
                int col = tidn + wn * RTSN;
                Breg[wn] = Bsub[buf][B_IDX(k, col)];
            }

            for (int wm = 0; wm < WPTM; wm++) {
                int row = tidm + wm * RTSM;
                Areg = Asub[buf][A_IDX(k, row)];

                for (int wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // =====================================
    // 🔹 Store results
    // =====================================
    for (int wm = 0; wm < WPTM; wm++) {

        int globalRow = offsetM + tidm + wm * RTSM;

        for (int wn = 0; wn < WPTN; wn++) {

            int globalCol = offsetN + tidn + wn * RTSN;

            C[globalRow * N + globalCol] = acc[wm][wn];
        }
    }
}
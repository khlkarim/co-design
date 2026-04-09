/*
 * kernel_best.cl
 * Optimized SGEMM — C = A * B (row-major)
 * TSK=23, double-buffering, prefetching enabled, no padding
 */

#define WIDTH  4          /* float4 vector width          */
#define TSM    128        /* tile size along M            */
#define TSN    128        /* tile size along N            */
#define TSK    23         /* tile depth                   */
#define WPTM   8          /* work per thread M-dim        */
#define WPTN   8          /* work per thread N-dim        */

#define RTSM   (TSM / WPTM)
#define RTSN   (TSN / WPTN)
#define LPTA   ((TSK * TSM) / (RTSM * RTSN))
#define LPTB   ((TSK * TSN) / (RTSM * RTSN))

#define floatX float4

/* Local memory layout without padding */
#define PADM   TSM
#define PADN   TSN
#define A_IDX(k, m)  ((k) * PADM + (m))
#define B_IDX(k, n)  ((k) * PADN + (n))

__kernel __attribute__((reqd_work_group_size(RTSM, RTSN, 1)))
void gemm_kernel_best(
    const int M,
    const int N,
    const int K,
    const int M_orig,
    const int N_orig,
    const __global floatX* restrict A,
    const __global floatX* restrict B,
    __global float* restrict C)
{
    const int tidm    = get_local_id(0);
    const int tidn    = get_local_id(1);
    const int offsetM = get_group_id(0) * TSM;
    const int offsetN = get_group_id(1) * TSN;
    const int flatTid = tidn * RTSM + tidm;

    __local float Asub[2][TSK * PADM];
    __local float Bsub[2][TSK * PADN];

    float acc[WPTM][WPTN];
    #pragma unroll
    for(int wm=0; wm<WPTM; wm++)
        #pragma unroll
        for(int wn=0; wn<WPTN; wn++)
            acc[wm][wn] = 0.0f;

    const int numTiles = K / TSK;

    /* Load tile 0 */
    #pragma unroll
    for(int la=0; la<LPTA/WIDTH; la++){
        int id = la*RTSM*RTSN + flatTid;
        int rowA = id % (TSM/WIDTH);
        int colA = id / (TSM/WIDTH);
        floatX vecA = A[colA*(M/WIDTH) + (offsetM/WIDTH) + rowA];
        vstore4(vecA, 0, &Asub[0][A_IDX(colA,rowA*WIDTH)]);
    }

    #pragma unroll
    for(int lb=0; lb<LPTB/WIDTH; lb++){
        int id = lb*RTSM*RTSN + flatTid;
        int rowB = id % (TSN/WIDTH);
        int colB = id / (TSN/WIDTH);
        floatX vecB = B[colB*(N/WIDTH) + (offsetN/WIDTH) + rowB];
        vstore4(vecB, 0, &Bsub[0][B_IDX(colB,rowB*WIDTH)]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Main tile loop with double-buffering */
    for(int t=0; t<numTiles; t++){
        const int curBuf = t & 1;
        const int nxtBuf = 1 - curBuf;
        const int tileK = TSK*(t+1);

        if(t+1<numTiles){
            #pragma unroll
            for(int la=0; la<LPTA/WIDTH; la++){
                int id = la*RTSM*RTSN + flatTid;
                int rowA = id % (TSM/WIDTH);
                int colA = id / (TSM/WIDTH);
                floatX vecA = A[(tileK+colA)*(M/WIDTH) + (offsetM/WIDTH) + rowA];
                vstore4(vecA, 0, &Asub[nxtBuf][A_IDX(colA,rowA*WIDTH)]);
            }
            #pragma unroll
            for(int lb=0; lb<LPTB/WIDTH; lb++){
                int id = lb*RTSM*RTSN + flatTid;
                int rowB = id % (TSN/WIDTH);
                int colB = id / (TSN/WIDTH);
                floatX vecB = B[(tileK+colB)*(N/WIDTH) + (offsetN/WIDTH) + rowB];
                vstore4(vecB, 0, &Bsub[nxtBuf][B_IDX(colB,rowB*WIDTH)]);
            }
        }

        /* Compute */
        #pragma unroll
        for(int k=0; k<TSK; k++){
            float Breg[WPTN];
            #pragma unroll
            for(int wn=0; wn<WPTN; wn++)
                Breg[wn] = Bsub[curBuf][B_IDX(k, tidn + wn*RTSN)];
            #pragma unroll
            for(int wm=0; wm<WPTM; wm++){
                float Areg = Asub[curBuf][A_IDX(k, tidm + wm*RTSM)];
                #pragma unroll
                for(int wn=0; wn<WPTN; wn++)
                    acc[wm][wn] = mad(Areg, Breg[wn], acc[wm][wn]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Write results */
    #pragma unroll
    for(int wm=0; wm<WPTM; wm++){
        int globalRow = offsetM + tidm + wm*RTSM;
        if(globalRow >= M_orig) continue;
        #pragma unroll
        for(int wn=0; wn<WPTN; wn++){
            int globalCol = offsetN + tidn + wn*RTSN;
            if(globalCol >= N_orig) continue;
            C[globalRow*N_orig + globalCol] = acc[wm][wn];
        }
    }
}
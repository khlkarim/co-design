#define WIDTH 4
#define TSM 128
#define TSN 128
#define TSK 16
#define WPTM 8
#define WPTN 8

#define RTSM (TSM / WPTM)
#define RTSN (TSN / WPTN)
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#define LPTB ((TSK*TSN)/(RTSM*RTSN))

#define floatX float4
#define PADM (TSM + 1)
#define PADN (TSN + 1)

#define A_IDX(k,m) ((k)*PADM + (m))
#define B_IDX(k,n) ((k)*PADN + (n))

__kernel __attribute__((reqd_work_group_size(RTSM, RTSN, 1)))
void gemm_kernel_best(
    const int M, const int N, const int K,
    const int M_orig, const int N_orig,
    const __global floatX* restrict A,
    const __global floatX* restrict B,
    __global float* restrict C)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int offsetM = get_group_id(0) * TSM;
    const int offsetN = get_group_id(1) * TSN;
    const int flatTid = tidn*RTSM + tidm;

    __local float Asub[2][TSK*PADM];
    __local float Bsub[2][TSK*PADN];

    float acc[WPTM][WPTN];
    #pragma unroll
    for(int wm=0; wm<WPTM; wm++)
        #pragma unroll
        for(int wn=0; wn<WPTN; wn++)
            acc[wm][wn]=0.0f;

    const int numTiles = K / TSK;

    // Load first tile
    #pragma unroll
    for(int la=0; la<LPTA/WIDTH; la++){
        int id = la*RTSM*RTSN + flatTid;
        int rowA = id % (TSM/WIDTH);
        int colA = id / (TSM/WIDTH);
        floatX vecA = A[colA*(M/WIDTH) + (offsetM/WIDTH) + rowA];
        vstore4(vecA, 0, &Asub[0][A_IDX(colA, rowA*WIDTH)]);

        int rowB = id % (TSN/WIDTH);
        int colB = id / (TSN/WIDTH);
        floatX vecB = B[(offsetN/WIDTH + rowB)*(K/WIDTH) + colB];
        vstore4(vecB, 0, &Bsub[0][B_IDX(colB, rowB*WIDTH)]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int t=0; t<numTiles; t++){
        int curBuf = t%2;
        int nxtBuf = (t+1)%2;

        // Prefetch next tile
        if(t+1<numTiles){
            #pragma unroll
            for(int la=0; la<LPTA/WIDTH; la++){
                int id = la*RTSM*RTSN + flatTid;

                int rowA = id % (TSM/WIDTH);
                int colA = id / (TSM/WIDTH);
                floatX vecA = A[(TSK*(t+1)+colA)*(M/WIDTH)+(offsetM/WIDTH)+rowA];
                vstore4(vecA, 0, &Asub[nxtBuf][A_IDX(colA, rowA*WIDTH)]);

                int rowB = id % (TSN/WIDTH);
                int colB = id / (TSN/WIDTH);
                floatX vecB = B[(offsetN/WIDTH+colB)*(K/WIDTH)+(TSK*(t+1)+rowB)];
                vstore4(vecB,0,&Bsub[nxtBuf][B_IDX(colB,rowB*WIDTH)]);
            }
        }

        // Compute
        for(int k=0; k<TSK; k++){
            float Breg[WPTN];
            for(int wn=0; wn<WPTN; wn++){
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[curBuf][B_IDX(k,col)];
            }

            float Areg;
            for(int wm=0; wm<WPTM; wm++){
                int row = tidm + wm*RTSM;
                Areg = Asub[curBuf][A_IDX(k,row)];
                #pragma unroll
                for(int wn=0; wn<WPTN; wn++)
                    acc[wm][wn] += Areg * Breg[wn];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store
    #pragma unroll
    for(int wm=0; wm<WPTM; wm++){
        int globalRow = offsetM + tidm + wm*RTSM;
        if(globalRow>=M_orig) continue;
        for(int wn=0; wn<WPTN; wn++){
            int globalCol = offsetN + tidn + wn*RTSN;
            if(globalCol>=N_orig) continue;
            C[globalRow*N_orig+globalCol] = acc[wm][wn];
        }
    }
}
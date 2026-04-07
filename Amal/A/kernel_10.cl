#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WIDTH 4          // float4
typedef float4 floatX;

#define TSM 32           // Tile size in M
#define TSN 32           // Tile size in N
#define TSK 16           // Tile size in K
#define WPTM 4           // Work per thread in M
#define WPTN 4           // Work per thread in N

#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)

__kernel void mmul(
    const int M_XL, const int N_XL, const int K_XL,
    const __global floatX* A,
    const __global floatX* B,
    __global float* C)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int groupM = get_group_id(0);
    const int groupN = get_group_id(1);
    const int offsetM = groupM * TSM;
    const int offsetN = groupN * TSN;

    __local float Asub[2][TSK * TSM];  // double-buffered
    __local float Bsub[2][TSK * TSN];

    float acc[WPTM][WPTN] = {0.0f};

    int numTiles = K_XL / TSK;

    for (int t = 0; t < numTiles; t++)
    {
        // Load tile of A
        for (int la = 0; la < (TSM*TSK)/(RTSM*RTSN*WIDTH); la++)
        {
            int id = tidn*RTSM + tidm + la*RTSM*RTSN;
            int row = id % (TSM/WIDTH);
            int col = id / (TSM/WIDTH);
            int globalRow = offsetM + row;
            int globalCol = t*TSK + col*WIDTH;
            floatX val = (globalRow < M_XL && globalCol < K_XL) ?
                         A[globalRow*K_XL/WIDTH + globalCol/WIDTH] : (floatX)(0.0f);
            ((floatX*)Asub[t%2])[row*TSK + col] = val;
        }

        // Load tile of B
        for (int lb = 0; lb < (TSN*TSK)/(RTSM*RTSN*WIDTH); lb++)
        {
            int id = tidn*RTSM + tidm + lb*RTSM*RTSN;
            int row = id % (TSK);
            int col = id / (TSK);
            int globalRow = t*TSK + row;
            int globalCol = offsetN + col*WIDTH;
            floatX val = (globalRow < K_XL && globalCol < N_XL) ?
                         B[globalRow*N_XL/WIDTH + globalCol/WIDTH] : (floatX)(0.0f);
            ((floatX*)Bsub[t%2])[row*TSN + col] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply-accumulate
        for (int k = 0; k < TSK; k++)
        {
            float Breg[WPTN];
            for (int wn = 0; wn < WPTN; wn++)
                Breg[wn] = Bsub[t%2][k*TSN + tidn + wn*RTSN];

            for (int wm = 0; wm < WPTM; wm++)
            {
                float Areg = Asub[t%2][k*TSM + tidm + wm*RTSM];
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += Areg * Breg[wn];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results
    for (int wm = 0; wm < WPTM; wm++)
    {
        int globalRow = offsetM + tidm + wm*RTSM;
        if (globalRow < M_XL)
        {
            for (int wn = 0; wn < WPTN; wn++)
            {
                int globalCol = offsetN + tidn + wn*RTSN;
                if (globalCol < N_XL)
                    C[globalRow*N_XL + globalCol] = acc[wm][wn];
            }
        }
    }
}
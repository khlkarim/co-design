    #define TSM 64                 // The tile-size in dimension M
    #define TSN 64                 // The tile-size in dimension N
    #define TSK 32                 // The tile-size in dimension K
    #define WPTN 8                 // The work-per-thread in dimension N
    #define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
    #define LPT ((TSK)/(RTSN)) // The loads-per-thread for (number of cells to load / number of active threads per row)


    // Pre-transpose the input matrix B and use rectangular tiles
    __kernel void mmul(const int M, const int N, const int K,
                          const __global float* A,
                          const __global float* B,
                          __global float* C) {
        
        // Thread identifiers
        const int row = get_local_id(0); // Local row ID (max: TSM)
        const int col = get_local_id(1); // Local col ID (max: TSN/WPTN)
        const int globalRow = TSM*get_group_id(0) + row; // 0..M
        const int globalCol = TSN*get_group_id(1) + col; // 0..N
     
        // Local memory to fit a tile of A and B
        __local float Asub[TSK][TSM];
        __local float Bsub[TSN][TSK + 2];
     
        // Initialise the accumulation registers
        float acc[WPTN];
        for (int w=0; w<WPTN; w++) {
            acc[w] = 0.0f;
        }
        
        // Loop over all tiles
        int numTiles = K/TSK;
        for (int t=0; t<numTiles; t++) {
     
            // Load one tile of A and B into local memory
            for (int l=0; l<LPT; l++) {
                int tiledIndex = TSK*t + col + l*RTSN;
                int indexA = tiledIndex*M + TSM*get_group_id(0) + row;
                int indexB = tiledIndex*N + TSN*get_group_id(1) + row;
                Asub[col + l*RTSN][row] = A[indexA];
                Bsub[row][col + l*RTSN] = B[indexB];
           }
            
            // Synchronise to make sure the tile is loaded
            barrier(CLK_LOCAL_MEM_FENCE);
     
            // Perform the computation for a single tile
            for (int k=0; k<TSK; k++) {
                for (int w=0; w<WPTN; w++) {
                    acc[w] += Asub[k][row] * Bsub[col + w*RTSN][k];
                }
            }
     
            // Synchronise before loading the next tile
            barrier(CLK_LOCAL_MEM_FENCE);
        }
     
        // Store the final results in C
        for (int w=0; w<WPTN; w++) {
            C[(globalCol + w*RTSN)*M + globalRow] = acc[w];
        }
    }

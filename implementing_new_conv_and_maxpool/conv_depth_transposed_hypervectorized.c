void conv_depth_transposed_hypervectorized_0(int I[], int O[], int W[], int D, int Hin, int Win, int kernel_size, int stride, int padding, int thread_no, int NUMTHREADS){{
    int Hout = (Hin + 2*padding - kernel_size)/stride +1;
    int Wout = (Win + 2*padding - kernel_size)/stride +1;
    volatile int ioutstart, ioutend;
    schedule1d(0, Hout, thread_no, NUMTHREADS, &ioutstart, &ioutend);
    int iinstart = ioutstart*stride;
    __m256i filter00[DCONST0];
    __m256i filter01[DCONST0];
    __m256i filter02[DCONST0];
    __m256i filter10[DCONST0];
    __m256i filter11[DCONST0];
    __m256i filter12[DCONST0];
    __m256i filter20[DCONST0];
    __m256i filter21[DCONST0];
    __m256i filter22[DCONST0];
    for (int n = 0; n < DCONST0; n++){{filter00[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*0]);}}
    for (int n = 0; n < DCONST0; n++){{filter01[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*1]);}}
    for (int n = 0; n < DCONST0; n++){{filter02[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*2]);}}
    for (int n = 0; n < DCONST0; n++){{filter10[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*3]);}}
    for (int n = 0; n < DCONST0; n++){{filter11[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*4]);}}
    for (int n = 0; n < DCONST0; n++){{filter12[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*5]);}}
    for (int n = 0; n < DCONST0; n++){{filter20[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*6]);}}
    for (int n = 0; n < DCONST0; n++){{filter21[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*7]);}}
    for (int n = 0; n < DCONST0; n++){{filter22[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*8]);}}
    __m256i cur00[DCONST0];
    __m256i cur01[DCONST0];
    __m256i cur02[DCONST0];
    __m256i cur10[DCONST0];
    __m256i cur11[DCONST0];
    __m256i cur12[DCONST0];
    __m256i cur20[DCONST0];
    __m256i cur21[DCONST0];
    __m256i cur22[DCONST0];

    for (int i = ioutstart; i < ioutend; i++){{
        for (int j = 0; j < Wout; j++){{
            int fs = D*(j*stride + i*stride*(Win+2*padding));//<----------------------------------------------------------------------------------------------------------------------
            for (int n = 0; n < DCONST0; n++){{cur00[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*0 + fs]);}}
            for (int n = 0; n < DCONST0; n++){{cur01[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*1 + fs]);}}
            for (int n = 0; n < DCONST0; n++){{cur02[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*2 + fs]);}}

            for (int n = 0; n < DCONST0; n++){{cur10[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*0 + 1*(Win+2*padding)*D  + fs]);}}
            for (int n = 0; n < DCONST0; n++){{cur11[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*1 + 1*(Win+2*padding)*D  + fs]);}}
            for (int n = 0; n < DCONST0; n++){{cur12[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*2 + 1*(Win+2*padding)*D  + fs]);}}

            for (int n = 0; n < DCONST0; n++){{cur20[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*0 + 2*(Win+2*padding)*D  + fs]);}}
            for (int n = 0; n < DCONST0; n++){{cur21[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*1 + 2*(Win+2*padding)*D  + fs]);}}
            for (int n = 0; n < DCONST0; n++){{cur22[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + D*2 + 2*(Win+2*padding)*D  + fs]);}}
            
            for (int n = 0; n < DCONST0; n++){{cur00[n] = _mm256_mullo_epi32(cur00[n], filter00[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur01[n] = _mm256_mullo_epi32(cur01[n], filter01[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur02[n] = _mm256_mullo_epi32(cur02[n], filter02[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur10[n] = _mm256_mullo_epi32(cur10[n], filter10[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur11[n] = _mm256_mullo_epi32(cur11[n], filter11[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur12[n] = _mm256_mullo_epi32(cur12[n], filter12[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur20[n] = _mm256_mullo_epi32(cur20[n], filter20[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur21[n] = _mm256_mullo_epi32(cur21[n], filter21[n]);}}
            for (int n = 0; n < DCONST0; n++){{cur22[n] = _mm256_mullo_epi32(cur22[n], filter22[n]);}}

            for (int n = 0; n < DCONST0; n++){{
                cur00[n] = _mm256_add_epi32(cur00[n], cur01[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur02[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur10[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur11[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur12[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur20[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur21[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur22[n]);
            }}
            for (int n = 0; n < DCONST0; n++){{
                _mm256_storeu_si256((__m256i*)&O[(j + i*Wout)*D + n*8] , cur00[n]);
            }}
        }}
    }}
}}
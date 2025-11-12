
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define wfilename "w.txt"
#define Din 256
#define Hin 640
#define Win 640
#define kernel_size 3
#define stride 1
#define INARRLEN Din*Hin*Win
#define Dout Din
#define MAXFILTERDEPTH Din/8
#define Hout (Hin - kernel_size)/stride + 1 
#define Wout (Win - kernel_size)/stride + 1 
#define OUTARRLEN Dout*Hout*Wout

int A[INARRLEN];
int B[OUTARRLEN];
int C[OUTARRLEN];
int D[OUTARRLEN];
int E[OUTARRLEN];
int F[OUTARRLEN];
int G[OUTARRLEN];
int W[Din*kernel_size*kernel_size];






void transpose(int I[], int O[], int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            O[i + j*M] = I[j + i*N];
        }
    }
}




void conv2d_depth_transposed_subfilter(int I[], int O[], int W[]){
    memset(O, 0, OUTARRLEN*sizeof(int));
    int cur_subfilter[Din];
    for (int h = 0; h < kernel_size; h++){
        for (int w = 0; w < kernel_size; w++){
            int fs = w*Din + h*kernel_size*Din;
            memcpy(cur_subfilter, &W[w*Din + h*kernel_size*Din], Din*sizeof(int));
            for (int i = 0; i < Hout; i++){
                fs = Win*stride*Din + w*Din + h*kernel_size*Din;
                for (int j = 0; j < Wout; j++){
                    for (int d = 0; d < Din; d++){
                        O[j*Din + i*Wout*Din + d] += I[fs + d] * cur_subfilter[d];
                    }
                    fs += Din*stride;
                }

            }

        }
    }
}

void conv2d_depth_transposed_hypervectorized(int I[], int O[], int W[]){

    __m256i filter00[Din/8];
    __m256i filter01[Din/8];
    __m256i filter02[Din/8];
    __m256i filter10[Din/8];
    __m256i filter11[Din/8];
    __m256i filter12[Din/8];
    __m256i filter20[Din/8];
    __m256i filter21[Din/8];
    __m256i filter22[Din/8];
    __m256i cur00[Din/8];
    __m256i cur01[Din/8];
    __m256i cur02[Din/8];
    __m256i cur10[Din/8];
    __m256i cur11[Din/8];
    __m256i cur12[Din/8];
    __m256i cur20[Din/8];
    __m256i cur21[Din/8];
    __m256i cur22[Din/8];

    for (int n = 0; n < Din/8; n++){filter00[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*0]);}
    for (int n = 0; n < Din/8; n++){filter01[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*1]);}
    for (int n = 0; n < Din/8; n++){filter02[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*2]);}
    for (int n = 0; n < Din/8; n++){filter10[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*3]);}
    for (int n = 0; n < Din/8; n++){filter11[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*4]);}
    for (int n = 0; n < Din/8; n++){filter12[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*5]);}
    for (int n = 0; n < Din/8; n++){filter20[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*6]);}
    for (int n = 0; n < Din/8; n++){filter21[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*7]);}
    for (int n = 0; n < Din/8; n++){filter22[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + Din*8]);}


    for (int i = 0; i < Hout; i++){
        for (int j = 0; j < Wout; j++){
            int fs = Din*(j*stride + i*stride*Win);
            for (int n = 0; n < Din/8; n++){cur00[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*0 + fs]);}
            for (int n = 0; n < Din/8; n++){cur01[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*1 + fs]);}
            for (int n = 0; n < Din/8; n++){cur02[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*2 + fs]);}

            for (int n = 0; n < Din/8; n++){cur10[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*0 + 1*Wout*Din + fs]);}
            for (int n = 0; n < Din/8; n++){cur11[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*1 + 1*Wout*Din  + fs]);}
            for (int n = 0; n < Din/8; n++){cur12[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*2 + 1*Wout*Din  + fs]);}

            for (int n = 0; n < Din/8; n++){cur20[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*0 + 2*Wout*Din  + fs]);}
            for (int n = 0; n < Din/8; n++){cur21[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*1 + 2*Wout*Din  + fs]);}
            for (int n = 0; n < Din/8; n++){cur22[n] = _mm256_loadu_si256((__m256i*)&I[n*8 + Din*2 + 2*Wout*Din  + fs]);}

            for (int n = 0; n < Din/8; n++){cur00[n] = _mm256_mullo_epi32(cur00[n], filter00[n]);}
            for (int n = 0; n < Din/8; n++){cur01[n] = _mm256_mullo_epi32(cur00[n], filter01[n]);}
            for (int n = 0; n < Din/8; n++){cur02[n] = _mm256_mullo_epi32(cur00[n], filter02[n]);}
            for (int n = 0; n < Din/8; n++){cur10[n] = _mm256_mullo_epi32(cur00[n], filter10[n]);}
            for (int n = 0; n < Din/8; n++){cur11[n] = _mm256_mullo_epi32(cur00[n], filter11[n]);}
            for (int n = 0; n < Din/8; n++){cur12[n] = _mm256_mullo_epi32(cur00[n], filter12[n]);}
            for (int n = 0; n < Din/8; n++){cur20[n] = _mm256_mullo_epi32(cur00[n], filter20[n]);}
            for (int n = 0; n < Din/8; n++){cur21[n] = _mm256_mullo_epi32(cur00[n], filter21[n]);}
            for (int n = 0; n < Din/8; n++){cur22[n] = _mm256_mullo_epi32(cur00[n], filter22[n]);}

            for (int n = 0; n < Din/8; n++){
                cur00[n] = _mm256_add_epi32(cur00[n], cur01[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur02[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur10[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur11[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur12[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur20[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur21[n]);
                cur00[n] = _mm256_add_epi32(cur00[n], cur22[n]);
            }
            for (int n = 0; n < Din/8; n++){
                _mm256_storeu_si256((__m256i*)&O[(j + i*Wout)*Din + n*8] , cur00[n]);
            }
        }
    }
}









int main(){
    for (long int i = 0; i < INARRLEN; i++){A[i] = rand()%512 - 256;}
    FILE* wfile = fopen(wfilename, "r");
    for (int i = 0; i < Din*kernel_size*kernel_size; i++){fscanf(wfile, "%d\n", &W[i]);}
    fclose(wfile);
    int max;




    clock_t time_vbelovedpts_beg = clock();
    conv2d_depth_transposed_subfilter(A, F, W);
    clock_t time_vbelovedpts_end = clock();
    printf("time for conv2d depth version transposed and subfiltered: %f s\n", (double)(time_vbelovedpts_end - time_vbelovedpts_beg)/CLOCKS_PER_SEC);
    max = F[0];for (int i = 1; i < OUTARRLEN; i++){if (F[i] > max){max = F[i];}}
    FILE* garbagevbelovedipts = fopen("GBG/garbagevbelovedptts.gbg", "w");fprintf(garbagevbelovedipts, "%d", max);fclose(garbagevbelovedipts);

    clock_t time_vbelovedptshv_beg = clock();
    conv2d_depth_transposed_hypervectorized(A, G, W);
    clock_t time_vbelovedptshv_end = clock();
    printf("time for conv2d depth version transposed and hypervectorized: %f s\n", (double)(time_vbelovedptshv_end - time_vbelovedptshv_beg)/CLOCKS_PER_SEC);
    max = G[0];for (int i = 1; i < OUTARRLEN; i++){if (G[i] > max){max = G[i];}}
    FILE* garbagevbelovediptshv = fopen("GBG/garbagevbelovedpttshv.gbg", "w");fprintf(garbagevbelovediptshv, "%d", max);fclose(garbagevbelovediptshv);

    return 0;
}

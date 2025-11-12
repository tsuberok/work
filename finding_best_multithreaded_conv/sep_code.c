

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define wfilename "wsepadd.txt"
#define Din 64
#define H 640
#define W 640
#define n_filters 128
#define INARRLEN Din*H*W
#define Dout n_filters
#define OUTARRLEN Dout*H*W

int A[OUTARRLEN];
int B[OUTARRLEN];
int C[OUTARRLEN];
int D[OUTARRLEN];
int WEIGHT[Din*n_filters];

__m256i ALLFILTERS[n_filters][Din/8];


void transpose(int I[], int M, int N){
    int* ISDONE = (int*)calloc(M*N,sizeof(int));
    for (long int c = 0; c < M*N; c++){
        if (ISDONE[c] == 0){
            int running_c = c;
            while (1){
                int i = running_c/N;
                int j = running_c - i*N;
                // I[i, j] --> I[j, i]
                int new_c = j*M + i;
                if (new_c == c){
                    break;
                }
                int temp = I[new_c];
                //printf("running_c new_c = %d %d\n", running_c, new_c);
                //printf("i j = %d %d\n", i, j);
                //printf("j i = %d %d\n", j, i);
                I[new_c] = I[running_c];
                ISDONE[new_c] = 1;
                running_c = new_c;

            }
        }
    }
    free(ISDONE);
}

void conv_sep_transposed(int I[], int O[], int WEIGHT[]){
    //memset(O, 0, OUTARRLEN*sizeof(int));
    for (int p = 0; p < H*W; p++){
        for (int n = 0; n < n_filters; n++){
            int s = 0;
            for (int d = 0; d < Din; d++){
                s += WEIGHT[d + n*Din] * I[d + p*Din];
            }
            O[p*n_filters + n] = s;
        }
    }
}

void conv_sep_transposed_inplace(int I[], int WEIGHT[], int move_mem){
    if (move_mem){
        for (int i = H-1; i >= 0; i--){
            for (int j = W-1; j>=0; j--){
                long int cur_pos = i*j*Din;
                long int desired_pos = i*j*n_filters;
                memmove(&I[desired_pos], &I[cur_pos], Din*sizeof(int));
            }
        }
    }

    int scratchpad_output[n_filters];
    for (int p = 0; p < H*W; p++){
        for (int n = 0; n < n_filters; n++){
            int s = 0;
            for (int d = 0; d < Din; d++){
                s += WEIGHT[d + n*Din] * I[d + p*n_filters];
            }
            scratchpad_output[n] = s;
        }
        memcpy(&I[p*n_filters], scratchpad_output, n_filters*sizeof(int));
    }
}


void conv_sep_transposed_vect(int I[], int O[], int WEIGHT[]){


    /*
    __m256i filter0[Din/8];
    __m256i filter1[Din/8];
    __m256i filter2[Din/8];
    __m256i filter3[Din/8];
    __m256i filter4[Din/8];
    __m256i filter5[Din/8];
    __m256i filter6[Din/8];
    __m256i filter7[Din/8];
    __m256i filter8[Din/8];
    __m256i filter9[Din/8];
    __m256i filter10[Din/8];
    __m256i filter11[Din/8];
    __m256i filter12[Din/8];
    __m256i filter13[Din/8];
    __m256i filter14[Din/8];
    __m256i filter15[Din/8];
    __m256i filter16[Din/8];
    __m256i filter17[Din/8];
    __m256i filter18[Din/8];
    __m256i filter19[Din/8];
    __m256i filter20[Din/8];
    __m256i filter21[Din/8];
    __m256i filter22[Din/8];
    __m256i filter23[Din/8];
    __m256i filter24[Din/8];
    __m256i filter25[Din/8];
    __m256i filter26[Din/8];
    __m256i filter27[Din/8];
    __m256i filter28[Din/8];
    __m256i filter29[Din/8];
    __m256i filter30[Din/8];
    __m256i filter31[Din/8];
    __m256i filter32[Din/8];
    __m256i filter33[Din/8];
    __m256i filter34[Din/8];
    __m256i filter35[Din/8];
    __m256i filter36[Din/8];
    __m256i filter37[Din/8];
    __m256i filter38[Din/8];
    __m256i filter39[Din/8];
    __m256i filter40[Din/8];
    __m256i filter41[Din/8];
    __m256i filter42[Din/8];
    __m256i filter43[Din/8];
    __m256i filter44[Din/8];
    __m256i filter45[Din/8];
    __m256i filter46[Din/8];
    __m256i filter47[Din/8];
    __m256i filter48[Din/8];
    __m256i filter49[Din/8];
    __m256i filter50[Din/8];
    __m256i filter51[Din/8];
    __m256i filter52[Din/8];
    __m256i filter53[Din/8];
    __m256i filter54[Din/8];
    __m256i filter55[Din/8];
    __m256i filter56[Din/8];
    __m256i filter57[Din/8];
    __m256i filter58[Din/8];
    __m256i filter59[Din/8];
    __m256i filter60[Din/8];
    __m256i filter61[Din/8];
    __m256i filter62[Din/8];
    __m256i filter63[Din/8];
    __m256i filter64[Din/8];

    int n = 0;
    for (int ni = 0; ni < Din/8; ni++){filter0[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter1[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter2[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter3[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter4[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter5[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter6[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter7[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter8[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter9[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter10[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter11[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter12[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter13[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter14[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter15[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter16[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter17[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter18[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter19[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter20[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter21[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter22[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter23[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter24[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter25[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter26[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter27[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter28[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter29[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter30[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter31[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter32[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter33[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter34[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter35[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter36[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter37[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter38[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter39[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter40[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter41[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter42[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter43[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter44[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter45[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter46[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter47[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter48[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter49[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter50[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter51[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter52[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter53[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter54[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter55[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter56[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter57[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter58[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter59[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter60[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter61[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter62[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;
    for (int ni = 0; ni < Din/8; ni++){filter63[ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);}
    n += Din;

    */

    for (int n = 0; n < n_filters; n++){
        for (int ni = 0; ni < Din/8; ni++){
            ALLFILTERS[n][ni] = _mm256_loadu_si256((__m256i*)&WEIGHT[n*Din + ni*8]);
        }
    }

    __m256i cur_piece[Din/8];
    int accz[8];
    __m256i acc;
    __m256i cur;
    for (int p = 0; p < H*W; p++){
        acc = _mm256_loadu_si256((__m256i*)accz);
        for (int pi = 0; pi < Din/8; pi++){
            cur_piece[pi] = _mm256_loadu_si256((__m256i*)&I[p*Din + pi*8]);
        }
        int s;
        int* ptr_to_acc = (int*)&acc;

        for (int n = 0; n < n_filters; n++){
            for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], ALLFILTERS[n][ni]);acc = _mm256_add_epi32(acc, cur);}
            for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
            O[p*n_filters + n] = s;
            s = 0;
        }
        /*
        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter0[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 0] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter1[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 1] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter2[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 2] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter3[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 3] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter4[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 4] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter5[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 5] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter6[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 6] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter7[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 7] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter8[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 8] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter9[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 9] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter10[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 10] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter11[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 11] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter12[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 12] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter13[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 13] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter14[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 14] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter15[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 15] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter16[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 16] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter17[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 17] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter18[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 18] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter19[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 19] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter20[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 20] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter21[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 21] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter22[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 22] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter23[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 23] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter24[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 24] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter25[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 25] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter26[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 26] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter27[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 27] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter28[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 28] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter29[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 29] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter30[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 30] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter31[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 31] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter32[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 32] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter33[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 33] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter34[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 34] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter35[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 35] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter36[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 36] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter37[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 37] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter38[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 38] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter39[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 39] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter40[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 40] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter41[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 41] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter42[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 42] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter43[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 43] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter44[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 44] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter45[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 45] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter46[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 46] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter47[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 47] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter48[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 48] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter49[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 49] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter50[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 50] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter51[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 51] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter52[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 52] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter53[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 53] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter54[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 54] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter55[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 55] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter56[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 56] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter57[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 57] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter58[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 58] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter59[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 59] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter60[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 60] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter61[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 61] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter62[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 62] = s;s = 0;

        for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], filter63[ni]);acc = _mm256_add_epi32(acc, cur);}
        for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
        O[p*n_filters + 63] = s;s = 0;

        */

    }

}


void conv_sep_hypervect(int I[], int O[], int WEIGHT[]){



    int scratchpad_input[Din];
    __m256i cur_piece[Din/8];
    int accz[8];
    __m256i acc;
    __m256i cur;
    for (int i = 0; i < H; i++){
        for (int j = 0; j < W; j++){
            for (int d = 0; d < Din; d++){
                scratchpad_input[d] = I[j + W*i + d*H*W];
            }
            acc = _mm256_loadu_si256((__m256i*)accz);
            for (int pi = 0; pi < Din/8; pi++){
                cur_piece[pi] = _mm256_loadu_si256((__m256i*)&scratchpad_input[pi*8]);
            }
            int s;
            int* ptr_to_acc = (int*)&acc;

            for (int n = 0; n < n_filters; n++){
                for (int ni = 0; ni < Din/8; ni++){cur = _mm256_mullo_epi32(cur_piece[ni], ALLFILTERS[n][ni]);acc = _mm256_add_epi32(acc, cur);}
                for (int ska = 0; ska < 8; ska++){s += *(ptr_to_acc + ska);}
                O[j + i*W + n*H*W] = s;
                s = 0;
            }
        }
    }
}


int main(){
    for (long int i = 0; i < INARRLEN; i++){A[i] = rand()%512 - 256;}
    FILE* wfile = fopen(wfilename, "r");
    for (int i = 0; i < Din*n_filters; i++){fscanf(wfile, "%d\n", &WEIGHT[i]);}
    fclose(wfile);
    int max;




    clock_t time_vt_beg = clock();
    conv_sep_transposed(A, B, WEIGHT);
    clock_t time_vt_end = clock();
    printf("time for conv2d separable transposed: %f s\n", (double)(time_vt_end - time_vt_beg)/CLOCKS_PER_SEC);
    max = B[0];for (int i = 1; i < OUTARRLEN; i++){if (B[i] > max){max = B[i];}}
    FILE* garbagevt = fopen("GBG/garbaget.gbg", "w");fprintf(garbagevt, "%d", max);fclose(garbagevt);





    clock_t time_vtv_beg = clock();
    conv_sep_transposed_vect(A, C, WEIGHT);
    clock_t time_vtv_end = clock();
    printf("time for conv2d separable transposed vectorized: %f s\n", (double)(time_vtv_end - time_vtv_beg)/CLOCKS_PER_SEC);
    max = C[0];for (int i = 1; i < OUTARRLEN; i++){if (C[i] > max){max = C[i];}}
    FILE* garbagevtv = fopen("GBG/garbagetv.gbg", "w");fprintf(garbagevtv, "%d", max);fclose(garbagevtv);



    clock_t time_vv_beg = clock();
    conv_sep_hypervect(A, D, WEIGHT);
    clock_t time_vv_end = clock();
    printf("time for conv2d separable hypervectorized: %f s\n", (double)(time_vv_end - time_vv_beg)/CLOCKS_PER_SEC);
    max = D[0];for (int i = 1; i < OUTARRLEN; i++){if (D[i] > max){max = D[i];}}
    FILE* garbagevv = fopen("GBG/garbagevv.gbg", "w");fprintf(garbagevv, "%d", max);fclose(garbagevv);


    /*

    clock_t time_vti_beg = clock();
    conv_sep_transposed_inplace(A, WEIGHT, 1);
    clock_t time_vti_end = clock();
    printf("time for conv2d separable transposed inplace: %f s\n", (double)(time_vti_end - time_vti_beg)/CLOCKS_PER_SEC);
    max = A[0];for (int i = 1; i < OUTARRLEN; i++){if (A[i] > max){max = A[i];}}
    FILE* garbagevti = fopen("GBG/garbageti.gbg", "w");fprintf(garbagevti, "%d", max);fclose(garbagevti);

    */
    return 0;
}


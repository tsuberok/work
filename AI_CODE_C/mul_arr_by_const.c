#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#define ARRLEN 64*(416+2)*(416+2)
#define NUMITER 1000
#define THAN_NUM 5.2
float arr[ARRLEN];

int main(){
    printf("prog start\n");
    
    double total_time, total_time_new;
    clock_t st_time0, end_time0, st_time1, end_time1;
    
    const __m256 const_vec = _mm256_set1_ps(THAN_NUM);
    for (int it = 0; it < NUMITER; it++){
        memset(arr, 0.2, sizeof(arr));
        st_time0 = clock();
        for (long int i = 0; i < ARRLEN; i++){arr[i] = THAN_NUM*arr[i];}
        end_time0 = clock();
        total_time += (double)(end_time0 - st_time0);
        memset(arr, 0.2, sizeof(arr));
        float* v = arr;
        st_time1 = clock();
        for (long int i = 0; i < ARRLEN; i+=8, v+=8){
            __m256 cur = _mm256_loadu_ps(v);
            cur = _mm256_mul_ps(cur, const_vec);
            _mm256_store_ps(&arr[i], cur);
            
        }
        end_time1 = clock();
        total_time_new += (double)(end_time1 - st_time1);
    }
    
    total_time = total_time / CLOCKS_PER_SEC;
    total_time_new = total_time_new / CLOCKS_PER_SEC;
    printf("Num iter is %d arrlen is %d and it took %f seconds\n",NUMITER,ARRLEN, total_time);
    printf("Num iter is %d arrlen is %d and it took %f seconds\n",NUMITER,ARRLEN, total_time_new);
    printf("The speedup is %f %%\n", (total_time/total_time_new - 1)*100);
    
    


    



    

    
    //for (long int i = 0; i < ARRLEN; i++){printf("%f\n", arr[i]);}
    scanf("%d");
    return 0;
}
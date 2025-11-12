#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>

#define ARRLEN 64*640*640
#define NUMITER 10
float arr[ARRLEN];

typedef struct PTR_N_LEN {
    float* arr_st;
    long int len;
} PTR_N_LEN;

void* arrsum(void* arg){
    PTR_N_LEN uwrp = *(PTR_N_LEN*)arg;
    float* arr_ptr = uwrp.arr_st;
    long int len = uwrp.len;
    double s = 0;
    __m256 acc = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (long int i = 0; i < 8*(len/8); i+=8 ){
        __m256 cur = _mm256_load_ps(arr_ptr);
        acc = _mm256_add_ps(acc, cur);
        //s += *arr_ptr;
        arr_ptr += 8;
    }
    float* ptr_to_acc = (float*)&acc;
    for (int i = 0; i < 8; i++){
        s += *ptr_to_acc;
        ptr_to_acc += 1;
    }
    arr_ptr += 1;//????? do i need it?????????
    for (long int i = 8*(len/8); i < len; i++){
        s += *arr_ptr;
        arr_ptr += 1;
    }
    //printf("the sum is %f\n", s);
    return NULL;
}

int main(){
    srand(time(NULL));
    

    double time_one_th = 0;
    double time_two_th = 0;
    clock_t st, end;
    for (long int i = 0; i < ARRLEN; i++){arr[i] = ((double)rand())/ ((double)RAND_MAX);}
    st = clock();
    for (int i = 0; i < NUMITER; i++){
        
        PTR_N_LEN wtf;
        wtf.arr_st = &arr[0];
        wtf.len = ARRLEN;
        arrsum(&wtf);
        
    }
    end = clock();
    time_one_th += (double)(end - st);
    st = clock();
    for (int i = 0; i < NUMITER; i++){
        
        PTR_N_LEN arg0;
        arg0.arr_st = &arr[0];
        arg0.len = ARRLEN/2;
        pthread_t th0;
        pthread_create(&th0, NULL, arrsum, &arg0);
        PTR_N_LEN arg1;
        arg1.arr_st = &arr[ARRLEN/2];
        arg1.len = ARRLEN/2;
        arrsum(&arg1);
        pthread_join(th0, NULL);
    }
    end = clock();
    time_two_th += (double)(end - st);
    time_one_th = time_one_th/CLOCKS_PER_SEC;
    time_two_th = time_two_th/CLOCKS_PER_SEC;
    printf("NUMITER = %d\n", NUMITER);
    printf("Single threaded solution took %f s\n", time_one_th);
    printf("2 threaded solution took %f\n", time_two_th);

    scanf("%d");
    return 0;
}
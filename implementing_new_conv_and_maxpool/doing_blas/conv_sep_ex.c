#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

pthread_barrier_t ONE_BARRIER;



void schedule1d(int beg, int fin, int thread_no, int number_of_threads, int* op_start, int* op_end){
    if (fin-beg <= number_of_threads){
        if (thread_no < fin-beg){
            *op_start = thread_no;
            *op_end = thread_no + 1;
        }
        else{
            *op_start = 0;
            *op_end = 0;
        }
    }
    else{
        volatile int num_to_do = (fin - beg)/number_of_threads;
        volatile int os = num_to_do*thread_no;
        volatile int oe = os + num_to_do;

        //if (thread_no == number_of_threads - 1){oe = fin;}
        if ((fin - beg) % number_of_threads != 0){
            if (thread_no < (fin - beg) % number_of_threads){
                os += thread_no;
                oe += thread_no;
                oe += 1;
            }
            else{
                os += (fin - beg) % number_of_threads;
                oe += (fin - beg) % number_of_threads;
            }
        }
        *op_start = os;
        *op_end = oe;
    }
}

void transpose(int I[], int O[], int M, int N, int thread_no, int NUMTHREADS){
    volatile int istart, iend;
    schedule1d(0, M, thread_no, NUMTHREADS, &istart, &iend);
    for (int i = istart; i < iend; i++){
        for (int j = 0; j < N; j++){
            O[i + j*M] = I[j + i*N];
        }
    }

}

void conv_sep_transposed_inplace(int* restrict I, int* restrict WEIGHT, int D, int H, int W, int n_filters, int thread_no, int NUMTHREADS){
    int pstart, pend;
    schedule1d(0, H*W,  thread_no, NUMTHREADS, &pstart, &pend);
    
    int s = 0;
    int* scratchpad_output = (int*)malloc(n_filters*sizeof(int));

    for (int p = pstart; p < pend; p++){
        for (int n = 0; n < n_filters; n++){
            s = 0;

            for (int d = 0; d < D; d++){
                s += WEIGHT[d + n*D] * I[d + p*n_filters];
            }
            scratchpad_output[n] = s;
        }
        memcpy(&I[p*n_filters], scratchpad_output, n_filters*sizeof(int));
    }
    free(scratchpad_output);
}

struct {
    int thread_no;
    int NUMTHREADS;
    int out_channels;
    int D;
    int Hnp;
    int Wnp;
    int kernel_size;
    int stride;
    int padding;
    int* WEIGHT_SEP;
    int* I;
    int* It;
    uint16_t* isCoal;
} typedef thread_args;

void* thread_routine(void* args){
    thread_args* arguments = (thread_args*)args;
    int thread_no = arguments->thread_no;
    int NUMTHREADS = arguments->NUMTHREADS;
    int out_channels = arguments->out_channels;
    int D = arguments->D;
    int Hnp = arguments->Hnp;
    int Wnp = arguments->Wnp;
    int kernel_size = arguments->kernel_size;
    int stride = arguments->stride;
    int padding = arguments->padding;
    int* WEIGHT_SEP = arguments->WEIGHT_SEP;
    int* I = arguments->I;
    int* It = arguments->It;
    uint16_t* isCoal = arguments->isCoal;

    conv_sep_transposed_inplace(It,  WEIGHT_SEP,  D, Hnp, Wnp, out_channels,    thread_no,  NUMTHREADS);

}

int main(){
     printf("program start\n");
    clock_t st, end;
    double wasted_time = 0;
    FILE* f = NULL;
    st = clock();
    int  NUMTHREADS, out_channels, D, H, W, kernel_size, stride, padding;
    f = fopen("params.txt", "r");

    fscanf(f, "%d ", &NUMTHREADS);
    fscanf(f, "%d ", &out_channels);
    fscanf(f, "%d ", &D);
    fscanf(f, "%d ", &H);
    fscanf(f, "%d ", &W);
    fscanf(f, "%d ", &kernel_size);
    fscanf(f, "%d ", &stride);
    fscanf(f, "%d", &padding);
    fclose(f);


    int* input_pre_transposed = (int*)malloc(D*H*W*sizeof(int));
    int* A = (int*)malloc(D*H*W*sizeof(int));
    int* WEIGHT_SEP = (int*)malloc(out_channels*out_channels*sizeof(int));
    int* correct_answer_not_transposed = (int*)malloc(out_channels*H*W*sizeof(int));
    int* CORRECT_ANSWER = (int*)malloc(out_channels*H*W*sizeof(int));
    f = fopen("inp.txt", "r");
    for (int i = 0; i < D*H*W; i++){
        fscanf(f, "%d\n", &input_pre_transposed[i]);
    }
    fclose(f);
    transpose(input_pre_transposed, A, D, H*W, 0, 1);
    f = fopen("weight_sep.txt", "r");
    for (int i = 0; i < D*D*1*1; i++){
        fscanf(f, "%d\n", &WEIGHT_SEP[i]);
    }
    fclose(f);
    f = fopen("outp.txt", "r");
    for (int i = 0; i < out_channels*H*W; i++){
        fscanf(f, "%d\n", &correct_answer_not_transposed[i]);
    }
    fclose(f);
    transpose(correct_answer_not_transposed, CORRECT_ANSWER, out_channels, H*W, 0, 1);

    pthread_barrier_init(&ONE_BARRIER, NULL, NUMTHREADS);
    thread_args* THARGS = (thread_args*)malloc(NUMTHREADS*sizeof(thread_args));
    for (int i = 0; i < NUMTHREADS; i++){
        THARGS[i].thread_no = i;
        THARGS[i].NUMTHREADS = NUMTHREADS;
        THARGS[i].D = D;
        THARGS[i].out_channels = out_channels;
        THARGS[i].Hnp = H;
        THARGS[i].Wnp = W;
        THARGS[i].kernel_size = kernel_size;
        THARGS[i].stride = stride;
        THARGS[i].padding = padding;
        THARGS[i].WEIGHT_SEP = WEIGHT_SEP;
        THARGS[i].I = input_pre_transposed;
        THARGS[i].It = A;
    }
    pthread_t* THREADS = (pthread_t*)malloc(NUMTHREADS*sizeof(pthread_t));
    end = clock();
    wasted_time += (double)(end - st)/CLOCKS_PER_SEC;
    for (int i = 0; i < NUMTHREADS; i++){        
        pthread_create(&THREADS[i], NULL, thread_routine, &THARGS[i]);
    }

    for (int i = 0; i < NUMTHREADS; i++){
        pthread_join(THREADS[i], NULL);
    }
    st = clock();
    int allCorrect = 1;
    for (int i = 0; i < out_channels*H*W; i++){
        if (abs(CORRECT_ANSWER[i] - A[i]) > 0){
            allCorrect = 0;
            printf("INCORRECT, %d %d\n", CORRECT_ANSWER[i], A[i]);
            //break;
        }
    }
    if (allCorrect){printf("all is correct!\n");}
    free(input_pre_transposed);
    free(A);
    free(WEIGHT_SEP);
    free(correct_answer_not_transposed);
    free(CORRECT_ANSWER);
    free(THARGS);
    end = clock();
    wasted_time += (double)(end - st)/CLOCKS_PER_SEC;
    printf("time wasted on reading and writing from memory = %fs\n", wasted_time);
    printf("programs end!\n");
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include<unistd.h>

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



void conv_sep_transposed_inplace_same_dim(int* restrict I, int* restrict WEIGHT, int D, int H, int W, int thread_no, int NUMTHREADS){
    int hstart, hend;
    schedule1d(0, H, thread_no, NUMTHREADS, &hstart, &hend);
    int s = 0;
    int n_filters = D;
    int* scratchpad_output = (int*)malloc(W*n_filters*sizeof(int));
    for (int i = hstart; i < hend; i++){
        for (int p = 0; p < W; p++){
            for (int n = 0; n < n_filters; n++){
                s = 0;
                for (int d = 0; d < D; d++){
                    s += WEIGHT[n*D + d] * I[(i*W + p)*n_filters + d];
                }
                scratchpad_output[p*n_filters + n] = s;
            }
        }
        //memcpy(&I[i*W*n_filters], scratchpad_output, n_filters*W*sizeof(int));
        for (int j = 0; j < n_filters*W; j++){I[i*W*n_filters +j] = scratchpad_output[j];}
    }
    /*
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
    */
    free(scratchpad_output);
}

struct {
    int NOTINPLACE;
    int thread_no;
    int NUMTHREADS;
    int D;
    int Hnp;
    int Wnp;

    int* WEIGHT_SEP;
    int* I;
    int* It;

} typedef thread_args;

void* thread_routine(void* args){
    thread_args* arguments = (thread_args*)args;
    int thread_no = arguments->thread_no;
    int NUMTHREADS = arguments->NUMTHREADS;
    int D = arguments->D;
    int Hnp = arguments->Hnp;
    int Wnp = arguments->Wnp;
    int* WEIGHT_SEP = arguments->WEIGHT_SEP;
    int* I = arguments->I;
    int* It = arguments->It;

    conv_sep_transposed_inplace_same_dim(It, WEIGHT_SEP, D,  Hnp,  Wnp, thread_no, NUMTHREADS);

}

int main(){
    printf("program start\n");
    clock_t st, end;
    double wasted_time = 0;
    FILE* f = NULL;
    st = clock();
    int  NUMTHREADS,  D, H, W;
    f = fopen("sep_params.txt", "r");

    fscanf(f, "%d ", &NUMTHREADS);
    fscanf(f, "%d ", &D);
    fscanf(f, "%d ", &H);
    fscanf(f, "%d ", &W);
    fclose(f);


    int* input_pre_transposed = (int*)malloc(D*H*W*sizeof(int));
    int* A = (int*)malloc(D*H*W*sizeof(int));

    int* WEIGHT_SEP = (int*)malloc(D*D*sizeof(int));
    int* correct_answer_not_transposed = (int*)malloc(D*H*W*sizeof(int));
    int* CORRECT_ANSWER = (int*)malloc(D*H*W*sizeof(int));
    f = fopen("input_sep.txt", "r");
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
    f = fopen("output_sep.txt", "r");
    for (int i = 0; i < D*H*W; i++){
        fscanf(f, "%d\n", &correct_answer_not_transposed[i]);
    }
    fclose(f);
    transpose(correct_answer_not_transposed, CORRECT_ANSWER, D, H*W, 0, 1);

    pthread_barrier_init(&ONE_BARRIER, NULL, NUMTHREADS);
    thread_args* THARGS = (thread_args*)malloc(NUMTHREADS*sizeof(thread_args));
    for (int i = 0; i < NUMTHREADS; i++){
        THARGS[i].thread_no = i;
        THARGS[i].NUMTHREADS = NUMTHREADS;
        THARGS[i].D = D;
        THARGS[i].Hnp = H;
        THARGS[i].Wnp = W;
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
    for (int i = 0; i < D*H*W; i++){
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
}

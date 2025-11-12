
#include <cblas.h>
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

void transpose(float I[], float O[], int M, int N, int thread_no, int NUMTHREADS){
    volatile int istart, iend;
    schedule1d(0, M, thread_no, NUMTHREADS, &istart, &iend);
    for (int i = istart; i < iend; i++){
        for (int j = 0; j < N; j++){
            O[i + j*M] = I[j + i*N];
        }
    }

}

void conv_sep_transposed_inplace(float* restrict I, float* restrict O, float* restrict WEIGHT, int D, int H, int W, int n_filters, int thread_no, int NUMTHREADS){
    int nstart, nend;
    schedule1d(0, n_filters,  thread_no, NUMTHREADS, &nstart, &nend);
    
    float s = 0;
    float* scratchpad_output = (float*)malloc(n_filters*sizeof(float));

    // for (int p = pstart; p < pend; p++){
    //     // float x[] = {1, 2, 3, 4, 5, 6};
    // // float vec[] = {10, 11, 12};
    // // float y[2];
    // // cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 3, 1, x, 3, vec, 1, 0, y, 1);

    //     cblas_sgemv(CblasRowMajor, CblasNoTrans, n_filters, D, 1, WEIGHT, D, &I[p*n_filters], 1, 0, scratchpad_output, 1);

    //     memcpy(&I[p*n_filters], scratchpad_output, n_filters*sizeof(float));
    // }

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, H*W, D, D, 1, I, D, WEIGHT, D, 0, O, D);

    //now trying if I isn't transposed...
    float coef;
    memset(&O[nstart*H*W], 0, (nend - nstart)*H*W*sizeof(float));
    for (int n = nstart; n < nend; n++){
        for (int d = 0; d < D; d++){
            cblas_saxpy(H*W, WEIGHT[n*D + d], &I[d*H*W], 1, &O[n*H*W], 1);
        }
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
    float* WEIGHT_SEP;
    float* I;
    float* It;
    float* O;
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
    float* WEIGHT_SEP = arguments->WEIGHT_SEP;
    float* I = arguments->I;
    float* It = arguments->It;
    float* O = arguments->O;
    uint16_t* isCoal = arguments->isCoal;

    conv_sep_transposed_inplace(I, O, WEIGHT_SEP,  D, Hnp, Wnp, out_channels,    thread_no,  NUMTHREADS);//trying to do it not transposed

}

int main(){
    printf("blas program start\n");

    // double x[] = {1, 2, 3};
    // double coeff = 1.5;
    // int one = 1;
    // int n = 3;
    // cblas_dscal(n, coeff, x, one);
    // for (int i = 0; i < n; i++){printf("%f ", x[i]);}

    // float x[] = {1, 2, 3, 4, 5, 6};
    // float vec[] = {10, 11, 12};
    // float y[2];
    // cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 3, 1, x, 3, vec, 1, 0, y, 1);
    // printf("result y is:\n");
    // for (int i = 0; i < 2; i++){
    //     printf("%f\n", y[i]);
    // }

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


    float* input_pre_transposed = (float*)malloc(D*H*W*sizeof(float));
    float* A = (float*)malloc(D*H*W*sizeof(float));
    float* WEIGHT_SEP = (float*)malloc(out_channels*out_channels*sizeof(float));
    float* O = (float*)malloc(out_channels*H*W*sizeof(float));
    float* correct_answer_not_transposed = (float*)malloc(out_channels*H*W*sizeof(float));
    float* CORRECT_ANSWER = (float*)malloc(out_channels*H*W*sizeof(float));
    f = fopen("inp.txt", "r");
    for (int i = 0; i < D*H*W; i++){
        fscanf(f, "%f\n", &input_pre_transposed[i]);
    }
    fclose(f);
    transpose(input_pre_transposed, A, D, H*W, 0, 1);
    f = fopen("weight_sep.txt", "r");
    for (int i = 0; i < D*D*1*1; i++){
        fscanf(f, "%f\n", &WEIGHT_SEP[i]);
    }
    fclose(f);
    f = fopen("outp.txt", "r");
    for (int i = 0; i < out_channels*H*W; i++){
        fscanf(f, "%f\n", &correct_answer_not_transposed[i]);
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
        THARGS[i].O = O;
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
        if (fabsf(correct_answer_not_transposed[i] - O[i]) > 0){
            allCorrect = 0;
            printf("BLAS INCORRECT, %d %d\n", correct_answer_not_transposed[i], O[i]);//trying the not transposed thing......
            //break;
        }
    }
    if (allCorrect){printf("BLAS all is correct!\n");}
    free(input_pre_transposed);
    free(A);
    free(WEIGHT_SEP);
    free(O);
    free(correct_answer_not_transposed);
    free(CORRECT_ANSWER);
    free(THARGS);
    end = clock();
    wasted_time += (double)(end - st)/CLOCKS_PER_SEC;
    printf("BLAS time wasted on reading and writing from memory = %fs\n", wasted_time);
    printf("programs end!\n");
    return 0;
}
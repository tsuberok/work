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
         int num_to_do = (fin - beg)/number_of_threads;
         int os = num_to_do*thread_no;
         int oe = os + num_to_do;

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

void conv2d(float* restrict I,float* restrict  O,float* restrict  WEIGHT,int  D,int  out_channels,int Hnp,int Wnp,int  kernel_size,int  stride,int  padding,int  thread_no,int  NUMTHREADS ){
    //at first we have to do this transpose thing....
    int H = Hnp+2*padding;
    int W = Wnp+2*padding;
    int Hout = (H-kernel_size)/stride+1;
    int Wout = (W-kernel_size)/stride+1;
    float* Iresh = (float*)malloc(Hout*Wout*kernel_size*kernel_size*sizeof(float));
    clock_t st, end, stt, endt;
    double sgemm_time = 0;
    double trans_time = 0;
    for (int d = 0; d < D; d++){
        stt = clock();
        int reshc = 0;
        for (int iout = 0; iout < Hout; iout++){
            for (int jout = 0; jout < Wout; jout++){
                for (int h = 0; h < kernel_size; h++){
                    memcpy(&Iresh[reshc], &I[d*H*W + (iout*stride+h)*W+jout*stride], kernel_size*sizeof(float));
                    reshc += kernel_size;
                }
            }
        }
        endt = clock();
        trans_time += (double)(endt - stt);
        //printf("GOT TO SGEMM\n");
        st = clock();
        cblas_sgemm(CblasRowMajor, 
                    CblasNoTrans, 
                    CblasTrans, 
                    Hout*Wout, 
                    out_channels, 
                    kernel_size*kernel_size, 
                    1, 
                    Iresh, 
                    kernel_size*kernel_size, 
                    &WEIGHT[d*out_channels*kernel_size*kernel_size], 
                    kernel_size*kernel_size, 
                    1, 
                    O, 
                    out_channels);
        //printf("SGEMM DONE\n");
        end = clock();
        sgemm_time += (double)(end - st);
    }
    sgemm_time = sgemm_time/CLOCKS_PER_SEC;
    trans_time = trans_time/CLOCKS_PER_SEC;
    printf("TIME IN SGEMM = %f\n", sgemm_time);
    printf("TIME IN TRANS = %f\n", trans_time);
    free(Iresh);
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
    float* WEIGHT;
    float* I;
    float* O;

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
    float* WEIGHT = arguments->WEIGHT;
    float* I = arguments->I;
    float* O = arguments->O;

    conv2d( I,  O,  WEIGHT,  D,  out_channels, Hnp, Wnp,  kernel_size,  stride,  padding,  thread_no,  NUMTHREADS );

}



int main(){
     printf("program start\n");
    clock_t st, end;
    double wasted_time = 0;
    FILE* f = NULL;
    st = clock();
    int  NUMTHREADS, out_channels, D, H, W, kernel_size, stride, padding;
    f = fopen("conv_params.txt", "r");

    fscanf(f, "%d ", &NUMTHREADS);
    fscanf(f, "%d ", &out_channels);
    fscanf(f, "%d ", &D);
    fscanf(f, "%d ", &H);
    fscanf(f, "%d ", &W);
    fscanf(f, "%d ", &kernel_size);
    fscanf(f, "%d ", &stride);
    fscanf(f, "%d", &padding);
    fclose(f);


    int Hout = (H +2*padding - kernel_size)/stride+1;
    int Wout = (W +2*padding - kernel_size)/stride+1;
    float* A = (float*)malloc(D*(H+2*padding)*(W+2*padding)*sizeof(float));
    float* B = (float*)malloc(out_channels*Hout*Wout*sizeof(float));
    float* WEIGHT = (float*)malloc(out_channels*D*kernel_size*kernel_size*sizeof(float));
    float* WEIGHT_RESH = (float*)malloc(out_channels*D*kernel_size*kernel_size*sizeof(float));
    float* CORRECT_ANSWER = (float*)malloc(out_channels*Hout*Wout*sizeof(float));
    f = fopen("input.txt", "r");
    for (int i = 0; i < D*(H+2*padding)*(W+2*padding); i++){
        fscanf(f, "%f\n", &A[i]);
    }
    fclose(f);
    f = fopen("weight.txt", "r");
    for (int i = 0; i < out_channels*D*kernel_size*kernel_size; i++){
        fscanf(f, "%f\n", &WEIGHT[i]);
    }
    fclose(f);
    //here we reshape weight a little bit
    int chcounter = 0;
    for (int channel = 0; channel < D; channel++){
        for (int filter = 0; filter < out_channels; filter++){
            memcpy(&WEIGHT_RESH[chcounter*kernel_size*kernel_size], &WEIGHT[filter*D*kernel_size*kernel_size + channel*kernel_size*kernel_size], kernel_size*kernel_size*sizeof(float));
            chcounter += 1;
        }
    }
    f = fopen("output.txt", "r");
    for (int i = 0; i < out_channels*Hout*Wout; i++){
        fscanf(f, "%f\n", &CORRECT_ANSWER[i]);
    }
    fclose(f);

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
        THARGS[i].WEIGHT = WEIGHT_RESH;
        THARGS[i].I = A;
        THARGS[i].O = B;
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
    for (int i = 0; i < out_channels*Hout*Wout; i++){
        if (fabsf(CORRECT_ANSWER[i] - B[i]) > 0){
            allCorrect = 0;
            printf("INCORRECT, %f %f\n", CORRECT_ANSWER[i], B[i]);
            break;
        }
    }
    if (allCorrect){printf("all is correct!\n");}
    free(A);
    free(WEIGHT);
    free(CORRECT_ANSWER);
    free(THARGS);
    end = clock();
    wasted_time += (double)(end - st)/CLOCKS_PER_SEC;
    printf("time wasted on reading and writing from memory = %fs\n", wasted_time);
    printf("programs end!\n");
    return 0;
}
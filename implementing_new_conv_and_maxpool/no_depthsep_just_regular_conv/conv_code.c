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

void f000(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1-1-1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] - I[Win*stride*i + j*stride +1] - I[Win*stride*i + j*stride +2];
        }
    }
}

void f001(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1-1 0
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] - I[Win*stride*i + j*stride +1] +0;
        }
    }
}

void f002(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1 -1 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] - I[Win*stride*i + j*stride +1] + I[Win*stride*i + j*stride +2];
        }
    }
}

void f010(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1 0 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] +0 - I[Win*stride*i + j*stride +2];
        }
    }
}

void f011(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1 0 0
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] +0 +0;
        }
    }
}

void f012(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1 0 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] +0 + I[Win*stride*i + j*stride +2];
        }
    }
}

void f020(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1 1 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] + I[Win*stride*i + j*stride +1] - I[Win*stride*i + j*stride +2];
        }
    }
}

void f021(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1 1 0
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] + I[Win*stride*i + j*stride +1] +0;
        }
    }
}

void f022(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//-1 1 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += - I[Win*stride*i + j*stride +0] + I[Win*stride*i + j*stride +1] + I[Win*stride*i + j*stride +2];
        }
    }
}

void f100(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//0 -1 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += +0 - I[Win*stride*i + j*stride +1] - I[Win*stride*i + j*stride +2];
        }
    }
}

void f101(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//0 -1 0
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += 0 - I[Win*stride*i + j*stride +1] +0;
        }
    }
}

void f102(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//0 -1 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += 0 - I[Win*stride*i + j*stride +1] + I[Win*stride*i + j*stride +2];
        }
    }
}

void f110(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//0 0 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += 0+0  - I[Win*stride*i + j*stride +2];
        }
    }
}

void f111(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){// 0 0 0
    
}

void f112(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){// 0 0 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += 0 +0 + I[Win*stride*i + j*stride +2];
        }
    }
}

void f120(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){// 0 1 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += 0 + I[Win*stride*i + j*stride +1] - I[Win*stride*i + j*stride +2];
        }
    }
}

void f121(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//0 1 0
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += 0 + I[Win*stride*i + j*stride +1] +0;
        }
    }
}

void f122(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//0 1 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += 0 + I[Win*stride*i + j*stride +1] + I[Win*stride*i + j*stride +2];
        }
    }
}

void f200(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//1 -1 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] - I[Win*stride*i + j*stride +1] - I[Win*stride*i + j*stride +2];
        }
    }
}

void f201(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//1 -1 0
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] - I[Win*stride*i + j*stride +1] +0;
        }
    }
}

void f202(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){// 1 -1 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] - I[Win*stride*i + j*stride +1] + I[Win*stride*i + j*stride +2];
        }
    }
}

void f210(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//1 0 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] +0 - I[Win*stride*i + j*stride +2];
        }
    }
}

void f211(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//1 0 0 
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] +0 +0;
        }
    }
}

void f212(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){// 1 0 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] +0 + I[Win*stride*i + j*stride +2];
        }
    }
}

void f220(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//1 1 -1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] + I[Win*stride*i + j*stride +1] - I[Win*stride*i + j*stride +2];
        }
    }
}

void f221(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//1 1 0
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += +I[Win*stride*i + j*stride +0] + I[Win*stride*i + j*stride +1] +0;
        }
    }
}

void f222(int* restrict I, int* restrict O, int num_to_do_h, int num_to_do_w, int Win, int Wout, int stride){//1 1 1
    for (int i = 0; i < num_to_do_h; i++){
        for (int j = 0; j < num_to_do_w; j++){
            O[Wout*i + j] += + I[Win*stride*i + j*stride +0] + I[Win*stride*i + j*stride +1] + I[Win*stride*i + j*stride +2];
        }
    }
}


void conv1d(int* restrict I, int* restrict O, int* restrict SUBFILTER, int num_to_do_H, int num_to_do_W, int Win, int Wout, int stride){
    int the_num = 100*SUBFILTER[0] + 10*SUBFILTER[1] + 1*SUBFILTER[2];
    switch (the_num){
        case 0:
            f000( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 1:
            f001( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 2:
            f002( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 10:
            f010( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 11:
            f011( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 12:
            f012( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 20:
            f020( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 21:
            f021( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 22:
            f022( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 100:
            f100( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 101:
            f101( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 102:
            f102( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 110:
            f110( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 111:
            f111( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 112:
            f112( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 120:
            f120( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 121:
            f121( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 122:
            f122( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 200:
            f200( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 201:
            f201( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 202:
            f202( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 210:
            f210( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 211:
            f211( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 212:
            f212( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 220:
            f220( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 221:
            f221( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
        case 222:
            f222( I,  O,  num_to_do_H,  num_to_do_W,  Win,  Wout,  stride);
            break;
    }
}

void conv2d(int* restrict I, int* restrict O, int* restrict WEIGHT, int in_channels, int out_channels,int Hnp,int Wnp, int kernel_size, int stride, int padding, int thread_no, int NUMTHREADS ){
    int H = Hnp + 2*padding;
    int W = Wnp + 2*padding;
    int Hout = (Hnp +2*padding - kernel_size)/stride+1;
    int Wout = (Wnp +2*padding - kernel_size)/stride+1;
    int ioutstart, ioutend;
    schedule1d(0, Hout, thread_no, NUMTHREADS, &ioutstart, &ioutend);
    int startin, startout, filter_start;
    for (int filter = 0; filter < out_channels; filter++){
        for (int channel = 0; channel < in_channels; channel++){

            for (int h = 0; h < kernel_size; h++){
                startin = channel*H*W + (ioutstart*stride+h)*W;
                startout = filter*Hout*Wout + ioutstart*Wout;
                filter_start = filter*in_channels*kernel_size*kernel_size + channel*kernel_size*kernel_size + h*kernel_size;
                conv1d(&I[startin], &O[startout], &WEIGHT[filter_start], ioutend - ioutstart, Wout, W, Wout, stride);
            }

        }
    }
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
    int* WEIGHT;
    int* I;
    int* O;

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
    int* WEIGHT = arguments->WEIGHT;
    int* I = arguments->I;
    int* O = arguments->O;

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
    int* A = (int*)malloc(D*(H+2*padding)*(W+2*padding)*sizeof(int));
    int* B = (int*)malloc(out_channels*Hout*Wout*sizeof(int));
    int* WEIGHT = (int*)malloc(out_channels*D*kernel_size*kernel_size*sizeof(int));
    int* CORRECT_ANSWER = (int*)malloc(out_channels*Hout*Wout*sizeof(int));
    f = fopen("input.txt", "r");
    for (int i = 0; i < D*(H+2*padding)*(W+2*padding); i++){
        fscanf(f, "%d\n", &A[i]);
    }
    fclose(f);
    f = fopen("weight.txt", "r");
    for (int i = 0; i < out_channels*D*kernel_size*kernel_size; i++){
        fscanf(f, "%d\n", &WEIGHT[i]);
    }
    fclose(f);
    for (int i = 0; i < out_channels*D*kernel_size*kernel_size; i++){WEIGHT[i] += 1;}//to make a switch statement for subfiltering
    f = fopen("output.txt", "r");
    for (int i = 0; i < out_channels*Hout*Wout; i++){
        fscanf(f, "%d\n", &CORRECT_ANSWER[i]);
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
        THARGS[i].WEIGHT = WEIGHT;
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
        if (abs(CORRECT_ANSWER[i] - B[i]) > 0){
            allCorrect = 0;
            printf("INCORRECT, %d %d\n", CORRECT_ANSWER[i], B[i]);
            //break;
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
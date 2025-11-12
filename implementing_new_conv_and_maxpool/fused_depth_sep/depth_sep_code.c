
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

void conv1d( int* restrict I,  int* restrict O,  int* restrict W,  int D, int stride, int thread_no){


    for ( int d = 0; d < D; d++){            
        O[d] += I[d]*W[d];            
    }

}

struct {
    int max_size;
    int size_free;
    int size_occupied;
    int* free_cells;
    int* occupied_cells;
} typedef memory_bank;

void mb_push(int val, int arr[], int* size, int max_size){
    if (*size < max_size){
        *size += 1;
    }
    if (*size > 1){
        for (int i = *size-1; i > 0; i--){
            arr[i] = arr[i-1];
        }
    }
    arr[0] = val; 
}
int mb_pop(int arr[], int* size){
    int ret = arr[*size-1];
    *size -= 1;
    return ret;
}

int inindex_from_outindex(int iout, int stride, int padding){
    return iout*stride - padding;
}

struct {
    int Hout;
    int Wout;
    int ioutstart;
    int ioutend;
    int isHeader;
    int isFooter;
    int memamc;
    int headeramc;
    int footeramc;
    int header_start_i;
    int footer_start_i;
    int start_iin;
    int end_iin;
} typedef internal_params_of_inplace_conv2d;

internal_params_of_inplace_conv2d* calculate_internal_params(int Hin, int Win, int kernel_size, int stride, int padding, int thread_no, int NUMTHREADS){
    internal_params_of_inplace_conv2d* p = malloc(1*sizeof(internal_params_of_inplace_conv2d));
    p->Hout = (Hin + 2*padding - kernel_size)/stride +1;
    p->Wout = (Win + 2*padding - kernel_size)/stride +1;
    schedule1d(0, p->Hout, thread_no, NUMTHREADS, &p->ioutstart, &p->ioutend);
    p->memamc = 1 + padding/stride;
    p->headeramc = p->memamc - stride;
    p->footeramc = kernel_size - stride;
    p->isHeader = p->headeramc > 0 && thread_no > 0;
    p->isFooter = p->footeramc > 0 && thread_no < NUMTHREADS -1;
    p->header_start_i = inindex_from_outindex(p->ioutstart, stride, padding);
    p->footer_start_i = inindex_from_outindex(p->ioutend, stride, padding) + stride - 1;
    p->start_iin = inindex_from_outindex(p->ioutstart, stride,padding);
    if (p->start_iin < 0){p->start_iin = 0;}
    p->end_iin = inindex_from_outindex(p->ioutend, stride, padding) + kernel_size - 1;
    if (p->end_iin >= Hin){p->end_iin = Hin;}
    //printf("----------------------\n thread_no = %d\n Hout Wout = %d %d\n ioutstart ioutend = %d %d\n memamc = %d\n headeramc = %d\n footeramc = %d\n isHeader isFooter = %d %d\n header_start_i = %d\n footer_start_i = %d\n start_iin end_iin = %d %d\n---------------------\n", thread_no, p->Hout, p->Wout, p->ioutstart, p->ioutend, p->memamc, p->headeramc, p->footeramc, p->isHeader, p->isFooter, p->header_start_i, p->footer_start_i, p->start_iin, p->end_iin);
    return p;
}

void conv_depth_sep_transposed_inplace(int* restrict I, int* restrict WEIGHT_DEPTH, int* restrict WEIGHT_SEP, 
                                    uint32_t* restrict WDAND, uint32_t* restrict WDXOR, int* restrict WDPLUSONE,
                                    uint32_t* restrict WSEPAND, uint32_t* restrict WSEPXOR, int* restrict WSEPPLUSONE,
                                    int out_channels, int D, int Hin, int Win, int kernel_size, int stride, int padding, int thread_no, int NUMTHREADS){
    internal_params_of_inplace_conv2d* conv_params = calculate_internal_params( Hin,  Win,  kernel_size,  stride,  padding,  thread_no,  NUMTHREADS);
    int Hout = conv_params->Hout;
    int Wout = conv_params->Wout;
    int ioutstart = conv_params->ioutstart;
    int ioutend = conv_params->ioutend;
    int isHeader = conv_params->isHeader;
    int isFooter = conv_params->isFooter;
    int memamc = conv_params->memamc;
    int headeramc = conv_params->headeramc;
    int footeramc = conv_params->footeramc;
    int header_start_i = conv_params->header_start_i;
    int footer_start_i = conv_params->footer_start_i;
    int start_iin = conv_params->start_iin;
    int end_iin = conv_params->end_iin;
    free(conv_params);

     int*  header = NULL;
     int*  footer = NULL;
    if (isHeader){
        header = (int*)malloc(headeramc*D*Win*sizeof(int));
        memcpy(header, &I[header_start_i*Win*D], headeramc*D*Win*sizeof(int));
    }
    if (isFooter){
        footer = (int*)malloc(footeramc*D*Win*sizeof(int));
        memcpy(footer, &I[footer_start_i*D*Win], footeramc*D*Win*sizeof(int));
    }


    pthread_barrier_wait(&ONE_BARRIER);

     int*  scratchpad_output = (int*)calloc(memamc*D*Wout,sizeof(int));
     int* depth_result = (int*)calloc(D, sizeof(int));
    memory_bank mem_bank;
    mem_bank.max_size = memamc;
    mem_bank.size_free = 0;
    mem_bank.size_occupied = 0;
    mem_bank.free_cells = (int*)malloc(mem_bank.max_size*sizeof(int));
    mem_bank.occupied_cells = (int*)malloc(mem_bank.max_size*sizeof(int));
    for (int i = 0; i < mem_bank.max_size; i++){
        mb_push(i, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
    }


    //printf("got to the main loop of the fused conv\n");    
    int iin = inindex_from_outindex(ioutstart, stride, padding);
    int jin = inindex_from_outindex(0, stride, padding);
    int isDepthDone;
    int s;
    int iin_in_bounds, jin_in_bounds;
    int conv_depth_start, start_in_weight;
    for (int iout = ioutstart; iout < ioutend; iout++){
        int poss = mb_pop(mem_bank.free_cells, &mem_bank.size_free);
        mb_push(poss, mem_bank.occupied_cells, &mem_bank.size_occupied, mem_bank.max_size);
        memset(&scratchpad_output[poss*D*Wout], 0, D*Wout*sizeof(int));
        jin = inindex_from_outindex(0, stride, padding);
        for (int jout = 0; jout < Wout; jout++){
            //jin = inindex_from_outindex(jout, stride, padding);
            memset(depth_result, 0, D*sizeof(int));
            isDepthDone = 0;
            for (int h = 0; h < kernel_size; h++){
                for (int w = 0; w < kernel_size; w++){
                    iin_in_bounds = iin+h >= 0 && iin+h < Hin;
                    jin_in_bounds = jin+w >= 0 && jin+w < Win;
                    if (iin_in_bounds && jin_in_bounds){
                        isDepthDone = 0;
                        start_in_weight = (w + h*kernel_size)*D;
                        if (isHeader && iin+h >= header_start_i && iin+h < header_start_i + headeramc){
                            conv_depth_start = (jin+w+Win*(iin+h-header_start_i))*D;
                            for (int inconvc = 0; inconvc < D; inconvc++){
                                //depth_result[inconvc] += header[conv_depth_start + inconvc] * WEIGHT_DEPTH[start_in_weight + inconvc];
                                depth_result[inconvc] += ((header[conv_depth_start + inconvc] & WDAND[start_in_weight + inconvc]) ^ WDXOR[start_in_weight + inconvc]) + WDPLUSONE[start_in_weight + inconvc];
                            }
                            isDepthDone = 1;
                        }
                        if (isFooter && iin+h >= footer_start_i && iin+h < footer_start_i + footeramc){
                            conv_depth_start = (jin+w+Win*(iin+h-footer_start_i))*D;
                            for (int inconvc = 0; inconvc < D; inconvc++){
                                //depth_result[inconvc] += footer[conv_depth_start + inconvc] * WEIGHT_DEPTH[start_in_weight + inconvc];
                                depth_result[inconvc] += ((footer[conv_depth_start + inconvc] & WDAND[start_in_weight + inconvc]) ^ WDXOR[start_in_weight + inconvc]) + WDPLUSONE[start_in_weight + inconvc];
                            }
                            isDepthDone = 1;
                        }
                        if (!isDepthDone){
                            conv_depth_start = ((jin+w)+Win*(iin+h))*D;
                            for (int inconvc = 0; inconvc < D; inconvc++){
                                //depth_result[inconvc] += I[conv_depth_start + inconvc] * WEIGHT_DEPTH[start_in_weight + inconvc];
                                depth_result[inconvc] += ((I[conv_depth_start + inconvc] & WDAND[start_in_weight + inconvc]) ^ WDXOR[start_in_weight + inconvc]) + WDPLUSONE[start_in_weight + inconvc];
                            }
                            isDepthDone = 1;
                        }
                    }
                }
            }

            if (isDepthDone){
                for (int n = 0; n < out_channels; n++){
                    s = 0;
                    for (int d = 0; d < D; d++){
                        //s += depth_result[d]*WEIGHT_SEP[n*D + d];
                        s += ((depth_result[d] & WSEPAND[n*D + d]) ^ WSEPXOR[n*D + d]) + WSEPPLUSONE[n*D + d];
                    }
                    scratchpad_output[poss*out_channels*Wout + jout*out_channels + n] = s;
                }
            }

            jin += stride;
        }
        if (mem_bank.size_occupied == mem_bank.max_size){

            int poss_output = mb_pop(mem_bank.occupied_cells, &mem_bank.size_occupied);
            int stride_factor = 0;
            if (stride > 1){
                stride_factor = padding;
            }
            //printf("COPYING SUCH DATA:\n");
            //for (int suka = poss_output*out_channels*Wout; suka < poss_output*out_channels*Wout + out_channels*Wout; suka++){printf("%d\n", scratchpad_output[suka]);}
            memcpy(&I[(iin+stride_factor)*D*Win], &scratchpad_output[poss_output*out_channels*Wout], out_channels*Wout*sizeof(int));
            mb_push(poss_output, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
        }
        iin += stride; 
    }
    while(mem_bank.size_occupied > 0){
        int poss_output = mb_pop(mem_bank.occupied_cells, &mem_bank.size_occupied);
        int stride_factor = 0;
        if (stride > 1){
            stride_factor = padding;
        }
        memcpy(&I[(iin+stride_factor)*D*Win], &scratchpad_output[poss_output*out_channels*Wout], out_channels*Wout*sizeof(int));
        mb_push(poss_output, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
        iin += stride;
    }
    free(scratchpad_output);
    free(depth_result);
    //free(depth_result_unreduced);
    free(mem_bank.free_cells);
    free(mem_bank.occupied_cells);
    if (isHeader){free(header);}
    if (isFooter){free(footer);}
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
    int* WEIGHT_DEPTH;
    int* WEIGHT_SEP;
    int* I;
    int* It;
    uint16_t* isCoal;
    uint32_t* WDAND;
    uint32_t* WDXOR;
    int* WDPLUSONE;
    uint32_t* WSEPAND;
    uint32_t* WSEPXOR;
    int* WSEPPLUSONE;
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
    int* WEIGHT_DEPTH = arguments->WEIGHT_DEPTH;
    int* WEIGHT_SEP = arguments->WEIGHT_SEP;
    int* I = arguments->I;
    int* It = arguments->It;
    uint16_t* isCoal = arguments->isCoal;
    uint32_t* WDAND = arguments->WDAND;
    uint32_t* WDXOR = arguments->WDXOR;
    int* WDPLUSONE =  arguments->WDPLUSONE;
    uint32_t* WSEPAND = arguments->WSEPAND;
    uint32_t* WSEPXOR = arguments->WSEPXOR;
    int* WSEPPLUSONE = arguments->WSEPPLUSONE;

    conv_depth_sep_transposed_inplace(It, WEIGHT_DEPTH,  WEIGHT_SEP,
        WDAND,  WDXOR,  WDPLUSONE,
        WSEPAND,  WSEPXOR,  WSEPPLUSONE,
       out_channels, D, Hnp, Wnp,  kernel_size,  stride,  padding,  thread_no,  NUMTHREADS);

}

int main(){
    printf("program start\n");
    clock_t st, end;
    double wasted_time = 0;
    FILE* f = NULL;
    st = clock();
    int  NUMTHREADS, out_channels, D, H, W, kernel_size, stride, padding;
    f = fopen("depth_sep_params.txt", "r");

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
    int* weight_depth_pre_transposed = (int*)malloc(D*kernel_size*kernel_size*sizeof(int));
    int* WEIGHT_DEPTH = (int*)malloc(D*kernel_size*kernel_size*sizeof(int));
    int* WEIGHT_SEP = (int*)malloc(D*out_channels*sizeof(int));
    uint32_t* WDAND = (uint32_t*)malloc(D*kernel_size*kernel_size*sizeof(int));
    uint32_t* WDXOR = (uint32_t*)malloc(D*kernel_size*kernel_size*sizeof(int));
    int* WDPLUSONE = (int*)malloc(D*kernel_size*kernel_size*sizeof(int));
    uint32_t* WSEPAND = (uint32_t*)malloc(D*out_channels*sizeof(int));
    uint32_t* WSEPXOR = (uint32_t*)malloc(D*out_channels*sizeof(int));
    int* WSEPPLUSONE = (int*)malloc(D*out_channels*sizeof(int));
    int* correct_answer_not_transposed = (int*)malloc(out_channels*H*W*sizeof(int));
    int* CORRECT_ANSWER = (int*)malloc(out_channels*H*W*sizeof(int));

    f = fopen("input_depth_sep.txt", "r");
    for (int i = 0; i < D*H*W; i++){
        fscanf(f, "%d\n", &input_pre_transposed[i]);
    }
    fclose(f);
    transpose(input_pre_transposed, A, D, H*W, 0, 1);
    f = fopen("weight_depth.txt", "r");
    for (int i = 0; i < D*kernel_size*kernel_size; i++){
        fscanf(f, "%d\n", &weight_depth_pre_transposed[i]);
    }
    fclose(f);
    transpose(weight_depth_pre_transposed, WEIGHT_DEPTH, D,kernel_size*kernel_size , 0, 1);
    for (int i = 0; i < D*kernel_size*kernel_size; i++){
        if (WEIGHT_DEPTH[i] == -1){
            WDAND[i] = UINT32_MAX;
            WDXOR[i] = UINT32_MAX;
            WDPLUSONE[i] = 1;
        }
        if (WEIGHT_DEPTH[i] == 0){
            WDAND[i] = 0;
            WDXOR[i] = 0;
            WDPLUSONE[i] = 0;
        }
        if (WEIGHT_DEPTH[i] == 1){
            WDAND[i] = UINT32_MAX;//AND with 1111111...
            WDXOR[i] = 0;//XOR with 0
            WDPLUSONE[i] = 0;//+0
        }
    }
    f = fopen("weight_sep.txt", "r");
    for (int i = 0; i < D*D*1*1; i++){
        fscanf(f, "%d\n", &WEIGHT_SEP[i]);
    }
    fclose(f);
    for (int i = 0; i < D*out_channels; i++){
        if (WEIGHT_SEP[i] == -1){
            WSEPAND[i] = UINT32_MAX;
            WSEPXOR[i] = UINT32_MAX;
            WSEPPLUSONE[i] = 1;
        }
        if (WEIGHT_SEP[i] == 0){
            WSEPAND[i] = 0;
            WSEPXOR[i] = 0;
            WSEPPLUSONE[i] = 0;
        }
        if (WEIGHT_SEP[i] == 1){
            WSEPAND[i] = UINT32_MAX;//AND with 1111111...
            WSEPXOR[i] = 0;//XOR with 0
            WSEPPLUSONE[i] = 0;//+0
        }
    }
    f = fopen("output_depth_sep.txt", "r");
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
        THARGS[i].WEIGHT_DEPTH = WEIGHT_DEPTH;
        THARGS[i].WEIGHT_SEP = WEIGHT_SEP;
        THARGS[i].I = input_pre_transposed;
        THARGS[i].It = A;
        THARGS[i].WDAND = WDAND;
        THARGS[i].WDXOR = WDXOR;
        THARGS[i].WDPLUSONE = WDPLUSONE;
        THARGS[i].WSEPAND = WSEPAND;
        THARGS[i].WSEPXOR = WSEPXOR;
        THARGS[i].WSEPPLUSONE = WSEPPLUSONE;
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
            break;
        }
    }
    if (allCorrect){printf("all is correct!\n");}
    free(input_pre_transposed);
    free(A);
    free(weight_depth_pre_transposed);
    free(WEIGHT_DEPTH);
    free(WEIGHT_SEP);
    free(correct_answer_not_transposed);
    free(CORRECT_ANSWER);
    free(THARGS);
    end = clock();
    wasted_time += (double)(end - st)/CLOCKS_PER_SEC;
    printf("time wasted on reading and writing from memory = %fs\n", wasted_time);
    printf("programs end!\n");
}

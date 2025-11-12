
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#define DCONST0 8


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

void conv_depth_transposed(int* I, int* O, int* W, int D, int Hin, int Win, int kernel_size, int stride, int padding, int thread_no, int NUMTHREADS){
    int Hout = (Hin + 2*padding - kernel_size)/stride +1;
    int Wout = (Win + 2*padding - kernel_size)/stride +1;
    volatile int ioutstart, ioutend;
    schedule1d(0, Hout, thread_no, NUMTHREADS, &ioutstart, &ioutend);
    int iinstart = ioutstart*stride;
    int* cur_subfilter = (int*)malloc(D*sizeof(int));
    for (int h = 0; h < kernel_size; h++){
        for (int w = 0; w < kernel_size; w++){
            memcpy(cur_subfilter, &W[w*D + h*kernel_size*D], D*sizeof(int));

            for (int i = ioutstart; i < ioutend; i++){
                int fs = (iinstart + (i-ioutstart)*stride)*(Win+2*padding)*D + w*D + h*(Win+2*padding)*D;
                for (int j = 0; j < Wout; j++){
                    for (int d = 0; d < D; d++){
                        O[j*D + i*Wout*D + d] += I[fs + d] * cur_subfilter[d];
                    }
                    fs += D*stride;
                }
            }

        }
    }
    free(cur_subfilter);
}





//__attribute__((noinline))
void conv1d( int* restrict I,  int* restrict O,  int* restrict W, int numToDo, int D, int stride, int thread_no){
     int fsi = 0;
     int fso = 0;
    for ( int j = 0; j < numToDo; j++){
        for ( int d = 0; d < D; d++){            
            O[fso + d] += I[fsi + d]*W[d];            
        }
        fsi += D*stride;
        fso += D*1;
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
        for (int i = *size-1; i > 0; i--){//here lies the problem
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

int iin_from_iout(int iout, int stride, int padding){
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
    p->header_start_i = iin_from_iout(p->ioutstart, stride, padding);
    p->footer_start_i = iin_from_iout(p->ioutend, stride, padding) + stride - 1;
    p->start_iin = iin_from_iout(p->ioutstart, stride,padding);
    if (p->start_iin < 0){p->start_iin = 0;}
    p->end_iin = iin_from_iout(p->ioutend, stride, padding) + kernel_size - 1;
    if (p->end_iin >= Hin){p->end_iin = Hin;}
    //printf("----------------------\n thread_no = %d\n Hout Wout = %d %d\n ioutstart ioutend = %d %d\n memamc = %d\n headeramc = %d\n footeramc = %d\n isHeader isFooter = %d %d\n header_start_i = %d\n footer_start_i = %d\n start_iin end_iin = %d %d\n---------------------\n", thread_no, p->Hout, p->Wout, p->ioutstart, p->ioutend, p->memamc, p->headeramc, p->footeramc, p->isHeader, p->isFooter, p->header_start_i, p->footer_start_i, p->start_iin, p->end_iin);
    return p;
}

void coal_mem(int I[], int orig_starts[], int final_starts[], int seg_sizes[], uint16_t isCoal[], int l, int thread_no, int NUMTHREADS){ 
    int isReady = 0;
    int cstart, cend;
    schedule1d(0, l,  thread_no, NUMTHREADS, &cstart, &cend);


    while(!isReady){
        isReady = 1;
        for (int i = cstart; i < cend; i++){
            if (isCoal[i] == 0){
                int target_start = final_starts[i];

                int target_end = target_start + seg_sizes[i];
                int self_start = orig_starts[i];
                int allClear = 1;
                for (int suka = 0; suka < l; suka++){//here is the fucking problem!!!!!!!
                    int s_start = orig_starts[suka];
                    int s_end = orig_starts[suka] + seg_sizes[suka];
                    if ((s_start >= target_start && s_start < target_end) || 
                        (s_end > target_start && s_end <= target_end) ||
                        (s_start <= target_start && s_end >= target_end)){
                        if (isCoal[suka] == 0 && suka != i){
                            allClear = 0;
                            isReady = 0;
                            break;
                        }
                    }
                }
                if (allClear){
                    //printf("moving from I[%d] to I[%d] %d elements\n", self_start, target_start, seg_sizes[i]);
                    memmove(&I[target_start], &I[self_start], seg_sizes[i]*sizeof(int));
                    isCoal[i] = 1;
                }
            }

        }
    }
    printf("coal mem end\n");
}

void conv_depth_transposed_inplace(int I[], int W[], int D, int Hin, int Win, int kernel_size, int stride, int padding, int thread_no, int NUMTHREADS, uint16_t isCoal[]){
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
     int*  cur_subfilter = (int*)malloc(D*sizeof(int));
     int*  scratchpad_output = (int*)calloc(memamc*D*Wout,sizeof(int));
    memory_bank mem_bank;
    mem_bank.max_size = memamc;
    mem_bank.size_free = 0;
    mem_bank.size_occupied = 0;
    mem_bank.free_cells = (int*)malloc(mem_bank.max_size*sizeof(int));
    mem_bank.occupied_cells = (int*)malloc(mem_bank.max_size*sizeof(int));
    for (int i = 0; i < mem_bank.max_size; i++){
        mb_push(i, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
    }



    int iin = iin_from_iout(ioutstart, stride, padding);
    int outOfBounds;
    int num_to_do;

    for (int i = ioutstart; i < ioutend; i++){
        int poss = mb_pop(mem_bank.free_cells, &mem_bank.size_free);
        mb_push(poss, mem_bank.occupied_cells, &mem_bank.size_occupied, mem_bank.max_size);

        memset(&scratchpad_output[poss*D*Wout], 0, D*Wout*sizeof(int));
        for (int h = 0; h < kernel_size; h++){
            for (int w = 0; w < kernel_size; w++){
                outOfBounds = iin + h < 0 || iin + h >= Hin;
                if (!outOfBounds){
                    //memcpy(cur_subfilter, &W[(w + h*kernel_size)*D], D*sizeof(int));
                    num_to_do = Wout;// num to do and num to move are different!!!!! First we do and then we move!!! And if we do the last, we don't move
                    //like, one is number of convs and another is number of steps... 
                    int start_in_iin = w - padding;
                    int start_iout = 0;
                    if (w < padding){
                        int m = (int)ceil(((float)(padding-w))/stride);
                        start_in_iin = w+m*stride - padding;
                        start_iout = m;
                        num_to_do -= m;
                    }
                    else{

                        int num_to_move = num_to_do - 1;
                        int w_final = start_in_iin + num_to_move*stride;
                        if (w_final >= Win){
                            int m = (int)ceil(num_to_do - (float)(Win - start_in_iin)/stride);
                            num_to_do -= m;
                        }
                    }

                    int isConvDone = 0;
                    if (isHeader && iin+h >= header_start_i && iin+h < header_start_i + headeramc){
                        conv1d(&header[D*Win*(iin+h - header_start_i) + start_in_iin*D], &scratchpad_output[poss*D*Wout + start_iout*D], &W[(w + h*kernel_size)*D], num_to_do, D, stride, thread_no);
                        isConvDone = 1;                        
                    }
                    if (isFooter && iin+h >= footer_start_i && iin+h < footer_start_i + footeramc){
                        conv1d(&footer[D*Win*(iin+h - footer_start_i) + start_in_iin*D], &scratchpad_output[poss*D*Wout + start_iout*D], &W[(w + h*kernel_size)*D], num_to_do, D, stride, thread_no);
                        isConvDone = 1;
                    }
                    if (!isConvDone){
                        conv1d( &I[D*Win*(iin+h)+start_in_iin*D],  &scratchpad_output[poss*D*Wout + start_iout*D], &W[(w + h*kernel_size)*D], num_to_do, D, stride, thread_no);
                    }
                }
            }


        }
        if (mem_bank.size_occupied == mem_bank.max_size){

            int poss_output = mb_pop(mem_bank.occupied_cells, &mem_bank.size_occupied);
            int stride_factor = 0;
            if (stride > 1){
                stride_factor = padding;
            }
            memcpy(&I[(iin+stride_factor)*D*Win], &scratchpad_output[poss_output*D*Wout], D*Wout*sizeof(int));
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
        memcpy(&I[(iin+stride_factor)*D*Win], &scratchpad_output[poss_output*D*Wout], D*Wout*sizeof(int));
        mb_push(poss_output, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
        iin += stride;
    }
    free(scratchpad_output);
    free(cur_subfilter);
    free(mem_bank.free_cells);
    free(mem_bank.occupied_cells);
    if (isHeader){free(header);}
    if (isFooter){free(footer);}
    if (Wout < Win){
        memset(&isCoal[ioutstart], 0, (ioutend - ioutstart)*sizeof(uint16_t));
        if (thread_no == 0){isCoal[0] = 1;}
        int* segments = (int*)malloc(Hout*sizeof(int));
        int* seg_lens = (int*)malloc(Hout*sizeof(int));
        int* finals = (int*)malloc(Hout*sizeof(int));
        for (int i = 0; i < Hout; i++){
            segments[i] = i*stride*Win*D;
            seg_lens[i] = Wout*D;
            finals[i] = i*Wout*D;
        }
        pthread_barrier_wait(&ONE_BARRIER);
        coal_mem(I, segments, finals, seg_lens, isCoal, Hout,  thread_no,  NUMTHREADS);
        free(segments);
        free(seg_lens);
        free(finals);
    }
}






void conv_sep_transposed(int I[], int O[], int WEIGHT[], int D, int H, int W, int n_filters, int thread_no, int NUMTHREADS){
    int pstart, pend;
    schedule1d(0, H*W,  thread_no, NUMTHREADS, &pstart, &pend);
    int s = 0;

    for (int p = pstart; p < pend; p++){
        for (int n = 0; n < n_filters; n++){
            s = 0;
            for (int d = 0; d < D; d++){
                s += WEIGHT[d + n*D] * I[d + p*D];
            }
            O[p*n_filters + n] = s;
        }
    }
}

void conv_sep_transposed_inplace(int* restrict I, int* restrict WEIGHT, int D, int H, int W, int n_filters, uint16_t isCoal[], int thread_no, int NUMTHREADS){
    int pstart, pend;
    schedule1d(0, H*W,  thread_no, NUMTHREADS, &pstart, &pend);
    if (n_filters > D){
        memset(&isCoal[pstart], 0, (pend - pstart)*sizeof(uint16_t));
        if (thread_no == 0){isCoal[0] = 1;}
        int* segments = (int*)malloc(H*W*sizeof(int));
        int* seg_lens = (int*)malloc(H*W*sizeof(int));
        int* finals = (int*)malloc(H*W*sizeof(int));
        for (int i = 0; i < H*W; i++){
            segments[i] = i*D;
            seg_lens[i] = D;
            finals[i] = i*n_filters;
        }

        pthread_barrier_wait(&ONE_BARRIER);
        coal_mem(I, segments, finals, seg_lens, isCoal, H*W,  thread_no,  NUMTHREADS);
        free(segments);
        free(seg_lens);
        free(finals);
    }
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
    int NOTINPLACE;
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
    int* O;
    uint16_t* isCoal;
} typedef thread_args;

void* thread_routine(void* args){
    thread_args* arguments = (thread_args*)args;
    int NOTINPLACE = arguments->NOTINPLACE;
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
    int* O = arguments->O;
    uint16_t* isCoal = arguments->isCoal;
    int HforSep = (Hnp + 2*padding - kernel_size)/stride + 1;
    int WforSep = (Wnp + 2*padding - kernel_size)/stride + 1;
    if (NOTINPLACE){
        conv_depth_transposed(It, O, WEIGHT_DEPTH, D, Hnp, Wnp,  kernel_size,  stride, padding,  thread_no,  NUMTHREADS);
        pthread_barrier_wait(&ONE_BARRIER);
        conv_sep_transposed(O, It, WEIGHT_SEP,  D,  HforSep,  WforSep,  out_channels,  thread_no,  NUMTHREADS);
    }
    else{
        conv_depth_transposed_inplace(It, WEIGHT_DEPTH, D,  Hnp, Wnp,  kernel_size,  stride,  padding,  thread_no,  NUMTHREADS, isCoal);
        pthread_barrier_wait(&ONE_BARRIER);
        conv_sep_transposed_inplace(It, WEIGHT_SEP,  D,  HforSep,  WforSep, out_channels,  isCoal,  thread_no,  NUMTHREADS);
    }
}

int main(){
    printf("program start\n");
    clock_t st, end;
    double wasted_time = 0;
    FILE* f = NULL;
    //f.write(f'notinplace numthreads in_channels out_channels H W kernel_size stride padding')
    st = clock();
    int  NOTINPLACE, NUMTHREADS, out_channels, D, H, W,  kernel_size, stride, padding;
    f = fopen("depth_sep_params.txt", "r");
    fscanf(f, "%d ", &NOTINPLACE);
    fscanf(f, "%d ", &NUMTHREADS);
    fscanf(f, "%d ", &D);
    fscanf(f, "%d ", &out_channels);
    fscanf(f, "%d ", &H);
    fscanf(f, "%d ", &W);
    fscanf(f, "%d ", &kernel_size);
    fscanf(f, "%d ", &stride);
    fscanf(f, "%d", &padding);
    fclose(f);
    int sizeActivationPreTransposed = D*(W + 2*padding*NOTINPLACE)*(H + 2*padding*NOTINPLACE);
    int HforSep = (H + 2*padding - kernel_size)/stride + 1;
    int WforSep = (W + 2*padding - kernel_size)/stride + 1;
    int size0, size1, size2, MAXSIZE;
    size0 = sizeActivationPreTransposed;
    size1 = D*HforSep*WforSep;
    size2 = out_channels*HforSep*WforSep;
    MAXSIZE = size0;
    if (size1 > MAXSIZE){MAXSIZE = size1;}
    if (size2 > MAXSIZE){MAXSIZE = size2;}
    int* input_pre_transposed = (int*)malloc(sizeActivationPreTransposed*sizeof(int));
    int* A = (int*)malloc(MAXSIZE*sizeof(int));
    int* B = (int*)malloc(MAXSIZE*sizeof(int));
    int* weight_depth_not_transposed = (int*)malloc(D*kernel_size*kernel_size*sizeof(int));
    int* WEIGHT_DEPTH = (int*)malloc(D*kernel_size*kernel_size*sizeof(int));
    int* WEIGHT_SEP = (int*)malloc(D*out_channels*1*1*sizeof(int));
    int* correct_answer_not_transposed = (int*)malloc(size2*sizeof(int));
    int* CORRECT_ANSWER = (int*)malloc(size2*sizeof(int));
    f = fopen("input_depth_sep.txt", "r");
    for (int i = 0; i < sizeActivationPreTransposed; i++){
        fscanf(f, "%d\n", &input_pre_transposed[i]);
    }
    fclose(f);
    transpose(input_pre_transposed, A, D, (W + 2*padding*NOTINPLACE)*(H + 2*padding*NOTINPLACE), 0, 1);
    f = fopen("weight_depth.txt", "r");
    for (int i = 0; i < D*kernel_size*kernel_size; i++){
        fscanf(f, "%d\n", &weight_depth_not_transposed[i]);
    }
    fclose(f);
    transpose(weight_depth_not_transposed, WEIGHT_DEPTH, D, kernel_size*kernel_size, 0, 1);
    f = fopen("weight_sep.txt", "r");
    for (int i = 0; i < out_channels*D*1*1; i++){
        fscanf(f, "%d\n", &WEIGHT_SEP[i]);
    }
    fclose(f);
    f = fopen("output_depth_sep.txt", "r");
    for (int i = 0; i < size2; i++){
        fscanf(f, "%d\n", &correct_answer_not_transposed[i]);
    }
    fclose(f);
    transpose(correct_answer_not_transposed, CORRECT_ANSWER, out_channels, HforSep*WforSep, 0, 1);

    pthread_barrier_init(&ONE_BARRIER, NULL, NUMTHREADS);
    printf("HforSep WforSep = %d %d\n", HforSep, WforSep);
    uint16_t* isCoal = (uint16_t*)malloc(HforSep*WforSep * sizeof(uint16_t));
    thread_args* THARGS = (thread_args*)malloc(NUMTHREADS*sizeof(thread_args));
    for (int i = 0; i < NUMTHREADS; i++){
        THARGS[i].NOTINPLACE = NOTINPLACE;
        THARGS[i].thread_no = i;
        THARGS[i].NUMTHREADS = NUMTHREADS;
        THARGS[i].out_channels = out_channels;
        THARGS[i].D = D;
        THARGS[i].Hnp = H;
        THARGS[i].Wnp = W;
        THARGS[i].kernel_size = kernel_size;
        THARGS[i].stride = stride;
        THARGS[i].padding = padding;
        THARGS[i].WEIGHT_DEPTH = WEIGHT_DEPTH;
        THARGS[i].WEIGHT_SEP = WEIGHT_SEP;
        THARGS[i].I = input_pre_transposed;
        THARGS[i].It = A;
        THARGS[i].O = B;
        THARGS[i].isCoal = isCoal;
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
    for (int i = 0; i < size2; i++){
        if (abs(CORRECT_ANSWER[i] - A[i]) > 0){
            allCorrect = 0;
            printf("INCORRECT, %d %d\n", CORRECT_ANSWER[i], A[i]);
        }
    }
    if (allCorrect){printf("all is correct!\n");}
    free(input_pre_transposed);
    free(A);
    free(B);
    free(weight_depth_not_transposed);
    free(WEIGHT_DEPTH);
    free(WEIGHT_SEP);
    free(correct_answer_not_transposed);
    free(CORRECT_ANSWER);
    free(THARGS);
    free(isCoal);
    end = clock();
    wasted_time += (double)(end - st)/CLOCKS_PER_SEC;
    printf("time wasted on reading and writing from memory = %fs\n", wasted_time);
    printf("programs end!\n");
}

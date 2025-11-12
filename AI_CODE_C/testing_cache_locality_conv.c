#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define D 3
#define H 640
#define W 640
#define OUT_CHANNELS 64
#define KERNEL_SIZE 3
#define STRIDE 1
#define ARRLEN D*H*W
#define OUTARRLEN OUT_CHANNELS * ((H - KERNEL_SIZE)/STRIDE) * ((W - KERNEL_SIZE)/STRIDE)
#define NUMITER 10
float arr[ARRLEN];
float arr_out[OUTARRLEN];


float*** create_tensor_handle_float_3d(float arr[], int DD,int HH,int WW){
    float*** tensor = malloc(DD*sizeof(float**));
    int hCounter = 0;
    for (int d = 0; d < DD; d++){
        float** table = malloc(HH*sizeof(float*));
        for (int i = 0; i < HH; i++){
            table[i] = &arr[hCounter];
            hCounter += WW;
        }
        tensor[d] = table;
    }
    return tensor;
}

void bitconv_no_tensors(float I[], float O[]){
int i = 0;
int j = 0;
for (long int iout = 0; iout < 637; iout++){
for (long int jout = 0; jout < 637; jout++){
long int fs = j + i*640;
O[jout+iout*637+0] += -I[fs+1]-I[fs+641]-I[fs+642]-I[fs+1280]+I[fs+1281]-I[fs+409600]+I[fs+409601]+I[fs+409602]+I[fs+410242]-I[fs+410880]+I[fs+410881]+I[fs+819200]+I[fs+819201]-I[fs+819202]+I[fs+819841]-I[fs+820481]+I[fs+820482]+0;
O[jout+iout*637+405769] += -I[fs+1]+I[fs+2]+I[fs+640]+I[fs+642]+I[fs+1280]+I[fs+1281]-I[fs+1282]-I[fs+409601]+I[fs+410242]+I[fs+410881]+I[fs+819200]-I[fs+819201]+I[fs+819841]-I[fs+820480]-I[fs+820481]+0;
O[jout+iout*637+811538] += +I[fs+0]+I[fs+1]-I[fs+641]+I[fs+642]+I[fs+1280]-I[fs+1281]-I[fs+409600]-I[fs+409601]-I[fs+409602]-I[fs+410240]+I[fs+410241]+I[fs+410880]+I[fs+410882]-I[fs+819201]-I[fs+819202]-I[fs+819840]-I[fs+819841]+I[fs+819842]+I[fs+820480]+I[fs+820481]-I[fs+820482]+0;
j += 1;
}
j = 0;
i += 1;
}
}


void bitconv_half_tensor(float*** I, float O[]){
int i = 0;
int j = 0;
for (long int iout = 0; iout < 637; iout++){
for (long int jout = 0; jout < 637; jout++){
O[jout+iout*637+0] += -I[0][i+0][j+1]-I[0][i+1][j+1]-I[0][i+1][j+2]-I[0][i+2][j+0]+I[0][i+2][j+1]-I[1][i+0][j+0]+I[1][i+0][j+1]+I[1][i+0][j+2]+I[1][i+1][j+2]-I[1][i+2][j+0]+I[1][i+2][j+1]+I[2][i+0][j+0]+I[2][i+0][j+1]-I[2][i+0][j+2]+I[2][i+1][j+1]-I[2][i+2][j+1]+I[2][i+2][j+2]+0;
O[jout+iout*637+405769] += -I[0][i+0][j+1]+I[0][i+0][j+2]+I[0][i+1][j+0]+I[0][i+1][j+2]+I[0][i+2][j+0]+I[0][i+2][j+1]-I[0][i+2][j+2]-I[1][i+0][j+1]+I[1][i+1][j+2]+I[1][i+2][j+1]+I[2][i+0][j+0]-I[2][i+0][j+1]+I[2][i+1][j+1]-I[2][i+2][j+0]-I[2][i+2][j+1]+0;
O[jout+iout*637+811538] += +I[0][i+0][j+0]+I[0][i+0][j+1]-I[0][i+1][j+1]+I[0][i+1][j+2]+I[0][i+2][j+0]-I[0][i+2][j+1]-I[1][i+0][j+0]-I[1][i+0][j+1]-I[1][i+0][j+2]-I[1][i+1][j+0]+I[1][i+1][j+1]+I[1][i+2][j+0]+I[1][i+2][j+2]-I[2][i+0][j+1]-I[2][i+0][j+2]-I[2][i+1][j+0]-I[2][i+1][j+1]+I[2][i+1][j+2]+I[2][i+2][j+0]+I[2][i+2][j+1]-I[2][i+2][j+2]+0;
j += 1;
}
j = 0;
i += 1;
}
}


void bitconv_all_tensors(float*** I, float*** O){
int i = 0;
int j = 0;
for (long int iout = 0; iout < 637; iout++){
for (long int jout = 0; jout < 637; jout++){
O[0][iout][jout] += -I[0][i+0][j+1]-I[0][i+1][j+1]-I[0][i+1][j+2]-I[0][i+2][j+0]+I[0][i+2][j+1]-I[1][i+0][j+0]+I[1][i+0][j+1]+I[1][i+0][j+2]+I[1][i+1][j+2]-I[1][i+2][j+0]+I[1][i+2][j+1]+I[2][i+0][j+0]+I[2][i+0][j+1]-I[2][i+0][j+2]+I[2][i+1][j+1]-I[2][i+2][j+1]+I[2][i+2][j+2]+0;
O[1][iout][jout] += -I[0][i+0][j+1]+I[0][i+0][j+2]+I[0][i+1][j+0]+I[0][i+1][j+2]+I[0][i+2][j+0]+I[0][i+2][j+1]-I[0][i+2][j+2]-I[1][i+0][j+1]+I[1][i+1][j+2]+I[1][i+2][j+1]+I[2][i+0][j+0]-I[2][i+0][j+1]+I[2][i+1][j+1]-I[2][i+2][j+0]-I[2][i+2][j+1]+0;
O[2][iout][jout] += +I[0][i+0][j+0]+I[0][i+0][j+1]-I[0][i+1][j+1]+I[0][i+1][j+2]+I[0][i+2][j+0]-I[0][i+2][j+1]-I[1][i+0][j+0]-I[1][i+0][j+1]-I[1][i+0][j+2]-I[1][i+1][j+0]+I[1][i+1][j+1]+I[1][i+2][j+0]+I[1][i+2][j+2]-I[2][i+0][j+1]-I[2][i+0][j+2]-I[2][i+1][j+0]-I[2][i+1][j+1]+I[2][i+1][j+2]+I[2][i+2][j+0]+I[2][i+2][j+1]-I[2][i+2][j+2]+0;
j += 1;
}
j = 0;
i += 1;
}
}


int main(){
    srand(time(NULL));
    float*** tensor_in = create_tensor_handle_float_3d(arr, D, H, W);
    float*** tensor_out = create_tensor_handle_float_3d(arr_out, OUT_CHANNELS,(H - KERNEL_SIZE)/STRIDE,(W - KERNEL_SIZE)/STRIDE);
    
    double time_no_tens = 0;
    double time_half_tens = 0;
    double time_tens = 0;
    clock_t st, end;
    for (int n = 0; n < NUMITER; n++){
        for (long int i = 0; i < ARRLEN; i++){arr[i] = ((double)rand())/ ((double)RAND_MAX);}
        memset(arr_out, 0, OUTARRLEN*sizeof(float));
        st = clock();
        bitconv_no_tensors(arr, arr_out);
        end = clock();
        time_no_tens += (double)(end - st);
        for (long int i = 0; i < ARRLEN; i++){arr[i] = ((double)rand())/ ((double)RAND_MAX);}
        memset(arr_out, 0, OUTARRLEN*sizeof(float));
        st = clock();
        //bitconv_half_tensor(tensor_in, arr_out);
        end = clock();
        time_half_tens += (double)(end - st);
        for (long int i = 0; i < ARRLEN; i++){arr[i] = ((double)rand())/ ((double)RAND_MAX);}
        memset(arr_out, 0, OUTARRLEN*sizeof(float));
        st = clock();
        //bitconv_all_tensors(tensor_in, tensor_out);
        end = clock();
        time_tens += (double)(end - st);
    }
    time_no_tens = time_no_tens/CLOCKS_PER_SEC;
    time_half_tens = time_half_tens/CLOCKS_PER_SEC;
    time_tens = time_tens/CLOCKS_PER_SEC;
    printf("NUMITER is %d and for no tensor it took %f s\n", NUMITER, time_no_tens);
    printf("NUMITER is %d and for half tensor(input) it took %f s\n", NUMITER, time_half_tens);
    printf("NUMITER is %d and for full tensor it took %f s\n", NUMITER, time_tens);

}

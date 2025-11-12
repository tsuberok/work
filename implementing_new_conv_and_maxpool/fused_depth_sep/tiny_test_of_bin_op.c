
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

int main(){
    int a, w, correct_ans, my_ans;
    FILE* f = fopen("testing_bin_op.txt", "r");
    fscanf(f, "%d ", &a);
    fscanf(f, "%d", &w);
    fclose(f);
    correct_ans = a*w;
    printf("a = %d. w = %d. Correct answer = %d\n", a, w, correct_ans);
    uint32_t wand, wxor;
    int wplusone;
    if (w == -1){
        wand = UINT32_MAX;
        wxor = UINT32_MAX;
        wplusone = 1;
    }
    if (w == 0){
        wand = 0;
        wxor = 0;
        wplusone = 0;
    }
    if (w == 1){
        wand = UINT32_MAX;
        wxor = 0;
        wplusone = 0;
    }
    my_ans = 0;
    //my_ans = (a & wand) ^ wxor;
    //my_ans += wplusone;
    my_ans += ((a & wand) ^ wxor) + wplusone;
    printf("binary op ans = %d\n", my_ans);
    return 0;
}

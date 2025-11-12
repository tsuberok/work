#include <stdio.h>
#include <stdlib.h>

int main()
{
    
    int arr_size = 5;
    int* b = (int*)malloc(arr_size*sizeof(int));
    for (int i = 0; i < arr_size; i++){
        b[i] = i;
    }
    
    for (int i = 0; i < arr_size; i++){
        b[i*2] += 10;//bug 2: compiles and runs with no warnings or crashes
    }

    return 0;
}
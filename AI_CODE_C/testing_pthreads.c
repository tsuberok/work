#include <stdio.h>
#include <pthread.h>
#define ARRLEN 10
float arr[ARRLEN] = {0.0};

void* foo(void* arg){
    for (int j = 0; j < ARRLEN/2; j++){
        arr[j] = 4.0;
    }
    return NULL;
}

int main(){
    pthread_t thread0;
    pthread_create(&thread0, NULL, foo, NULL);
    int join_ret = pthread_join(thread0, NULL);
    printf("join_ret = %d\n", join_ret);
    for (int i = 0; i < ARRLEN; i++){printf("%d %f\n", i, arr[i]);}
    scanf("%d");
    return 0;
}
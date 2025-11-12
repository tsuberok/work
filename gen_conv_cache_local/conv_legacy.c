#include <stdio.h>
#include <time.h>
#define INARRLEN 3072
#define OUTARRLEN 2700
long int iib(int d,int i,int j,int H,int W){return d*(H*W)+i*W+j;}
float I[INARRLEN];
float O[OUTARRLEN];
int main(){
FILE* input_file = fopen("input.txt", "r");
for (long int i = 0; i < INARRLEN; i++){fscanf(input_file, "%f", &I[i]);}
fclose(input_file);
clock_t st, end;
st = clock();

int i = 0;
int j = 0;
for (int iout = 0; iout < 30; iout++){
	for (int jout = 0; jout < 30; jout++){
		long int fs = j+32*i;O[iib(0,iout,jout,30,30)]+=-I[fs+0]-I[fs+1]+I[fs+2]+I[fs+32]+I[fs+33]-I[fs+34]-I[fs+65]+I[fs+66]+0;
O[iib(1,iout,jout,30,30)]+=-I[fs+1024]-I[fs+1025]-I[fs+1056]-I[fs+1058]-I[fs+1089]+0;
O[iib(2,iout,jout,30,30)]+=+I[fs+2048]-I[fs+2080]-I[fs+2082]-I[fs+2113]+0;

		j+=1;}
	i+=1;
	j=0;}
end = clock();
printf("Conv time = %f\n", ((double)(end - st))/CLOCKS_PER_SEC);
FILE* output_file = fopen("output_legacy.txt", "w");
for (long int i = 0; i < OUTARRLEN; i++){fprintf(output_file,"%f\n", O[i]);}
fclose(output_file);
return 0;
}

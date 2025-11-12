#include <stdio.h>
#include <time.h>
#define INARRLEN 3072
#define OUTARRLEN 2700
float I[INARRLEN];
float O[OUTARRLEN];
int main(){
FILE* input_file = fopen("input.txt", "r");
for (long int i = 0; i < INARRLEN; i++){fscanf(input_file, "%f", &I[i]);}
fclose(input_file);
clock_t st, end;
st = clock();
int i_0_0_0 = 0;
int i_0_0_1 = 32;
int i_0_0_2 = 64;
int i_1_0_0 = 1024;
int i_1_0_1 = 1056;
int i_1_0_2 = 1088;
int i_2_0_0 = 2048;
int i_2_0_1 = 2080;
int i_2_0_2 = 2112;
int o_0 = 0;
int o_1 = 900;
int o_2 = 1800;
for(long int i = 0; i < 900; i += 1){
	O[o_0]+=-I[i_0_0_0+0]-I[i_0_0_0+1]+I[i_0_0_0+2]+0;
	O[o_0]+=+I[i_0_0_1+0]+I[i_0_0_1+1]-I[i_0_0_1+2]+0;
	O[o_0]+=-I[i_0_0_2+1]+I[i_0_0_2+2]+0;
	O[o_1]+=-I[i_1_0_0+0]-I[i_1_0_0+1]+0;
	O[o_1]+=-I[i_1_0_1+0]-I[i_1_0_1+2]+0;
	O[o_1]+=-I[i_1_0_2+1]+0;
	O[o_2]+=+I[i_2_0_0+0]+0;
	O[o_2]+=-I[i_2_0_1+0]-I[i_2_0_1+2]+0;
	O[o_2]+=-I[i_2_0_2+1]+0;
	i_0_0_0 += 1;
	i_0_0_1 += 1;
	i_0_0_2 += 1;
	i_1_0_0 += 1;
	i_1_0_1 += 1;
	i_1_0_2 += 1;
	i_2_0_0 += 1;
	i_2_0_1 += 1;
	i_2_0_2 += 1;
	if ((i+1) % 30 == 0){
		i_0_0_0 += 2;
		i_0_0_1 += 2;
		i_0_0_2 += 2;
		i_1_0_0 += 2;
		i_1_0_1 += 2;
		i_1_0_2 += 2;
		i_2_0_0 += 2;
		i_2_0_1 += 2;
		i_2_0_2 += 2;
	}
	o_0 += 1;
	o_1 += 1;
	o_2 += 1;
}
end = clock();
printf("Conv time = %f\n", ((double)(end - st))/CLOCKS_PER_SEC);
FILE* output_file = fopen("output.txt", "w");
for (long int i = 0; i < OUTARRLEN; i++){fprintf(output_file,"%f\n", O[i]);}
fclose(output_file);
return 0;
}


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define wfilename "w.txt"
#define Din 256
#define Hin 640
#define Win 640
#define kernel_size 3
#define stride 1
#define INARRLEN Din*Hin*Win
#define Dout Din
#define Hout (Hin - kernel_size)/stride + 1 
#define Wout (Win - kernel_size)/stride + 1 
#define OUTARRLEN Dout*Hout*Wout

int A[INARRLEN];
int B[OUTARRLEN];
int W[Din*kernel_size*kernel_size];

void conv2d_depth_v1(int I[], int O[]){
    memset(O, 0, OUTARRLEN*sizeof(int));
    long int fs = 0;
    for (int i = 0; i < Hout; i++){
        for (int j = 0; j < Wout; j++){
            /*V1*/
            fs += stride;
        }
        fs = stride*Win;
    }
}

void conv2d_depth_v2(int I[], int O[]){
    memset(O, 0, OUTARRLEN*sizeof(int));
    long int fs = 0;
    fs = 0;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 0*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 0;
}
fs = 409600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 1*Wout*Hout] += -I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 409600;
}
fs = 819200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 2*Wout*Hout] += +I[fs + 0]+I[fs + 640]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 819200;
}
fs = 1228800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 3*Wout*Hout] += -I[fs + 0]+I[fs + 1]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 1228800;
}
fs = 1638400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 4*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 1638400;
}
fs = 2048000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 5*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 2048000;
}
fs = 2457600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 6*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 2457600;
}
fs = 2867200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 7*Wout*Hout] += -I[fs + 0]+I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 2867200;
}
fs = 3276800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 8*Wout*Hout] += +I[fs + 1]+I[fs + 2]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 3276800;
}
fs = 3686400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 9*Wout*Hout] += -I[fs + 1]+I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 3686400;
}
fs = 4096000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 10*Wout*Hout] += +I[fs + 0]-I[fs + 2]+I[fs + 641]+I[fs + 642]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 4096000;
}
fs = 4505600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 11*Wout*Hout] += +I[fs + 1]-I[fs + 2]-I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 4505600;
}
fs = 4915200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 12*Wout*Hout] += -I[fs + 2]-I[fs + 640]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 4915200;
}
fs = 5324800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 13*Wout*Hout] += -I[fs + 0]+I[fs + 640]+I[fs + 641]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 5324800;
}
fs = 5734400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 14*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 5734400;
}
fs = 6144000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 15*Wout*Hout] += -I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 6144000;
}
fs = 6553600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 16*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 641]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 6553600;
}
fs = 6963200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 17*Wout*Hout] += -I[fs + 0]+I[fs + 1]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 6963200;
}
fs = 7372800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 18*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 7372800;
}
fs = 7782400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 19*Wout*Hout] += -I[fs + 0]+I[fs + 641]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 7782400;
}
fs = 8192000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 20*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 641]+I[fs + 642]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 8192000;
}
fs = 8601600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 21*Wout*Hout] += -I[fs + 0]-I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 8601600;
}
fs = 9011200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 22*Wout*Hout] += -I[fs + 0]-I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 9011200;
}
fs = 9420800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 23*Wout*Hout] += +I[fs + 2]+I[fs + 640]-I[fs + 642]-I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 9420800;
}
fs = 9830400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 24*Wout*Hout] += -I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 9830400;
}
fs = 10240000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 25*Wout*Hout] += -I[fs + 641]-I[fs + 642]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 10240000;
}
fs = 10649600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 26*Wout*Hout] += +I[fs + 0]-I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 10649600;
}
fs = 11059200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 27*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 641]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 11059200;
}
fs = 11468800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 28*Wout*Hout] += -I[fs + 0]-I[fs + 2]+I[fs + 640]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 11468800;
}
fs = 11878400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 29*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 11878400;
}
fs = 12288000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 30*Wout*Hout] += +I[fs + 1]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 12288000;
}
fs = 12697600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 31*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 2]+I[fs + 641]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 12697600;
}
fs = 13107200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 32*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 13107200;
}
fs = 13516800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 33*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 13516800;
}
fs = 13926400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 34*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 13926400;
}
fs = 14336000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 35*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 14336000;
}
fs = 14745600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 36*Wout*Hout] += -I[fs + 1]-I[fs + 2]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 14745600;
}
fs = 15155200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 37*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 15155200;
}
fs = 15564800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 38*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 640]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 15564800;
}
fs = 15974400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 39*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 15974400;
}
fs = 16384000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 40*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 640]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 16384000;
}
fs = 16793600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 41*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 16793600;
}
fs = 17203200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 42*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 17203200;
}
fs = 17612800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 43*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 17612800;
}
fs = 18022400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 44*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]+I[fs + 641]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 18022400;
}
fs = 18432000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 45*Wout*Hout] += +I[fs + 0]+I[fs + 2]+I[fs + 641]-I[fs + 642]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 18432000;
}
fs = 18841600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 46*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 18841600;
}
fs = 19251200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 47*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 19251200;
}
fs = 19660800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 48*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 19660800;
}
fs = 20070400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 49*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 641]+I[fs + 642]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 20070400;
}
fs = 20480000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 50*Wout*Hout] += +I[fs + 0]+I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 20480000;
}
fs = 20889600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 51*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]+I[fs + 642]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 20889600;
}
fs = 21299200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 52*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 642]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 21299200;
}
fs = 21708800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 53*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 21708800;
}
fs = 22118400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 54*Wout*Hout] += +I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 22118400;
}
fs = 22528000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 55*Wout*Hout] += -I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 22528000;
}
fs = 22937600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 56*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 22937600;
}
fs = 23347200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 57*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 23347200;
}
fs = 23756800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 58*Wout*Hout] += +I[fs + 2]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 23756800;
}
fs = 24166400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 59*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 24166400;
}
fs = 24576000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 60*Wout*Hout] += -I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 24576000;
}
fs = 24985600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 61*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 24985600;
}
fs = 25395200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 62*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 25395200;
}
fs = 25804800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 63*Wout*Hout] += +I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 25804800;
}
fs = 26214400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 64*Wout*Hout] += +I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 26214400;
}
fs = 26624000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 65*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 26624000;
}
fs = 27033600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 66*Wout*Hout] += -I[fs + 0]-I[fs + 2]+I[fs + 640]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 27033600;
}
fs = 27443200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 67*Wout*Hout] += -I[fs + 0]+I[fs + 2]-I[fs + 640]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 27443200;
}
fs = 27852800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 68*Wout*Hout] += +I[fs + 1]+I[fs + 2]-I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 27852800;
}
fs = 28262400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 69*Wout*Hout] += -I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 28262400;
}
fs = 28672000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 70*Wout*Hout] += -I[fs + 1]+I[fs + 2]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 28672000;
}
fs = 29081600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 71*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 641]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 29081600;
}
fs = 29491200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 72*Wout*Hout] += +I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 29491200;
}
fs = 29900800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 73*Wout*Hout] += -I[fs + 2]-I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 29900800;
}
fs = 30310400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 74*Wout*Hout] += +I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 30310400;
}
fs = 30720000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 75*Wout*Hout] += -I[fs + 0]-I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 30720000;
}
fs = 31129600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 76*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 640]-I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 31129600;
}
fs = 31539200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 77*Wout*Hout] += +I[fs + 0]-I[fs + 2]-I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 31539200;
}
fs = 31948800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 78*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 31948800;
}
fs = 32358400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 79*Wout*Hout] += +I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 32358400;
}
fs = 32768000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 80*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 32768000;
}
fs = 33177600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 81*Wout*Hout] += +I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 33177600;
}
fs = 33587200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 82*Wout*Hout] += -I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 33587200;
}
fs = 33996800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 83*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 33996800;
}
fs = 34406400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 84*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 34406400;
}
fs = 34816000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 85*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 34816000;
}
fs = 35225600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 86*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 641]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 35225600;
}
fs = 35635200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 87*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 35635200;
}
fs = 36044800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 88*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 36044800;
}
fs = 36454400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 89*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 36454400;
}
fs = 36864000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 90*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 36864000;
}
fs = 37273600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 91*Wout*Hout] += +I[fs + 0]-I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 37273600;
}
fs = 37683200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 92*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 640]+I[fs + 641]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 37683200;
}
fs = 38092800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 93*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 38092800;
}
fs = 38502400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 94*Wout*Hout] += +I[fs + 1]+I[fs + 2]-I[fs + 641]+I[fs + 642]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 38502400;
}
fs = 38912000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 95*Wout*Hout] += +I[fs + 0]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 38912000;
}
fs = 39321600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 96*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 641]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 39321600;
}
fs = 39731200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 97*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 641]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 39731200;
}
fs = 40140800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 98*Wout*Hout] += -I[fs + 640]+I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 40140800;
}
fs = 40550400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 99*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 40550400;
}
fs = 40960000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 100*Wout*Hout] += -I[fs + 0]+I[fs + 2]+I[fs + 640]-I[fs + 642]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 40960000;
}
fs = 41369600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 101*Wout*Hout] += +I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 41369600;
}
fs = 41779200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 102*Wout*Hout] += -I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 41779200;
}
fs = 42188800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 103*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 42188800;
}
fs = 42598400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 104*Wout*Hout] += -I[fs + 0]-I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 42598400;
}
fs = 43008000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 105*Wout*Hout] += +I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 43008000;
}
fs = 43417600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 106*Wout*Hout] += -I[fs + 1]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 43417600;
}
fs = 43827200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 107*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 43827200;
}
fs = 44236800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 108*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 641]-I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 44236800;
}
fs = 44646400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 109*Wout*Hout] += -I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 44646400;
}
fs = 45056000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 110*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 642]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 45056000;
}
fs = 45465600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 111*Wout*Hout] += -I[fs + 0]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 45465600;
}
fs = 45875200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 112*Wout*Hout] += +I[fs + 0]+I[fs + 2]+I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 45875200;
}
fs = 46284800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 113*Wout*Hout] += +I[fs + 1]+I[fs + 2]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 46284800;
}
fs = 46694400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 114*Wout*Hout] += +I[fs + 1]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 46694400;
}
fs = 47104000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 115*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 47104000;
}
fs = 47513600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 116*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 47513600;
}
fs = 47923200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 117*Wout*Hout] += +I[fs + 1]+I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 47923200;
}
fs = 48332800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 118*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 640]+I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 48332800;
}
fs = 48742400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 119*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 640]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 48742400;
}
fs = 49152000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 120*Wout*Hout] += +I[fs + 1]-I[fs + 640]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 49152000;
}
fs = 49561600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 121*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 49561600;
}
fs = 49971200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 122*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 642]+0;
		fs += stride;
	}
	fs = stride*Win + 49971200;
}
fs = 50380800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 123*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 50380800;
}
fs = 50790400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 124*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 2]+0;
		fs += stride;
	}
	fs = stride*Win + 50790400;
}
fs = 51200000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 125*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]-I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 51200000;
}
fs = 51609600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 126*Wout*Hout] += +I[fs + 1]+I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 51609600;
}
fs = 52019200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 127*Wout*Hout] += -I[fs + 2]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 52019200;
}
fs = 52428800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 128*Wout*Hout] += +I[fs + 0]-I[fs + 641]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 52428800;
}
fs = 52838400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 129*Wout*Hout] += +I[fs + 0]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 52838400;
}
fs = 53248000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 130*Wout*Hout] += +I[fs + 0]+I[fs + 640]+I[fs + 641]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 53248000;
}
fs = 53657600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 131*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 53657600;
}
fs = 54067200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 132*Wout*Hout] += -I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 54067200;
}
fs = 54476800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 133*Wout*Hout] += +I[fs + 0]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 54476800;
}
fs = 54886400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 134*Wout*Hout] += -I[fs + 0]+I[fs + 2]-I[fs + 640]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 54886400;
}
fs = 55296000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 135*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 641]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 55296000;
}
fs = 55705600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 136*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 641]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 55705600;
}
fs = 56115200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 137*Wout*Hout] += +I[fs + 0]-I[fs + 640]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 56115200;
}
fs = 56524800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 138*Wout*Hout] += -I[fs + 0]-I[fs + 2]-I[fs + 640]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 56524800;
}
fs = 56934400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 139*Wout*Hout] += -I[fs + 0]-I[fs + 2]-I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 56934400;
}
fs = 57344000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 140*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 641]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 57344000;
}
fs = 57753600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 141*Wout*Hout] += -I[fs + 0]-I[fs + 2]-I[fs + 641]+I[fs + 642]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 57753600;
}
fs = 58163200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 142*Wout*Hout] += +I[fs + 0]+I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 58163200;
}
fs = 58572800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 143*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 58572800;
}
fs = 58982400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 144*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 58982400;
}
fs = 59392000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 145*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 59392000;
}
fs = 59801600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 146*Wout*Hout] += -I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 59801600;
}
fs = 60211200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 147*Wout*Hout] += -I[fs + 1]-I[fs + 2]+0;
		fs += stride;
	}
	fs = stride*Win + 60211200;
}
fs = 60620800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 148*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 60620800;
}
fs = 61030400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 149*Wout*Hout] += -I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 61030400;
}
fs = 61440000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 150*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 61440000;
}
fs = 61849600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 151*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 61849600;
}
fs = 62259200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 152*Wout*Hout] += -I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 62259200;
}
fs = 62668800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 153*Wout*Hout] += +I[fs + 1]-I[fs + 641]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 62668800;
}
fs = 63078400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 154*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 63078400;
}
fs = 63488000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 155*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 63488000;
}
fs = 63897600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 156*Wout*Hout] += -I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 63897600;
}
fs = 64307200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 157*Wout*Hout] += -I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 64307200;
}
fs = 64716800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 158*Wout*Hout] += +I[fs + 0]-I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 64716800;
}
fs = 65126400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 159*Wout*Hout] += +I[fs + 1]+I[fs + 2]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 65126400;
}
fs = 65536000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 160*Wout*Hout] += +I[fs + 0]-I[fs + 641]+I[fs + 642]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 65536000;
}
fs = 65945600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 161*Wout*Hout] += -I[fs + 0]+I[fs + 2]+I[fs + 640]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 65945600;
}
fs = 66355200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 162*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 66355200;
}
fs = 66764800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 163*Wout*Hout] += +I[fs + 1]+I[fs + 2]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 66764800;
}
fs = 67174400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 164*Wout*Hout] += +I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 67174400;
}
fs = 67584000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 165*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 67584000;
}
fs = 67993600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 166*Wout*Hout] += +I[fs + 0]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 67993600;
}
fs = 68403200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 167*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 641]-I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 68403200;
}
fs = 68812800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 168*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 640]+I[fs + 641]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 68812800;
}
fs = 69222400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 169*Wout*Hout] += +I[fs + 640]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 69222400;
}
fs = 69632000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 170*Wout*Hout] += -I[fs + 2]+I[fs + 640]+I[fs + 642]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 69632000;
}
fs = 70041600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 171*Wout*Hout] += -I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 70041600;
}
fs = 70451200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 172*Wout*Hout] += +I[fs + 1]+I[fs + 2]+I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 70451200;
}
fs = 70860800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 173*Wout*Hout] += -I[fs + 0]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 70860800;
}
fs = 71270400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 174*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]+I[fs + 641]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 71270400;
}
fs = 71680000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 175*Wout*Hout] += +I[fs + 2]+I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 71680000;
}
fs = 72089600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 176*Wout*Hout] += -I[fs + 0]+I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 72089600;
}
fs = 72499200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 177*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 640]+I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 72499200;
}
fs = 72908800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 178*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 642]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 72908800;
}
fs = 73318400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 179*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 73318400;
}
fs = 73728000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 180*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 73728000;
}
fs = 74137600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 181*Wout*Hout] += -I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 74137600;
}
fs = 74547200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 182*Wout*Hout] += -I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]+0;
		fs += stride;
	}
	fs = stride*Win + 74547200;
}
fs = 74956800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 183*Wout*Hout] += +I[fs + 1]-I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 74956800;
}
fs = 75366400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 184*Wout*Hout] += -I[fs + 0]-I[fs + 2]+I[fs + 640]-I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 75366400;
}
fs = 75776000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 185*Wout*Hout] += +I[fs + 0]+I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 75776000;
}
fs = 76185600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 186*Wout*Hout] += +I[fs + 0]+I[fs + 2]-I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 76185600;
}
fs = 76595200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 187*Wout*Hout] += -I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 76595200;
}
fs = 77004800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 188*Wout*Hout] += -I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 642]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 77004800;
}
fs = 77414400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 189*Wout*Hout] += +I[fs + 1]-I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 77414400;
}
fs = 77824000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 190*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 77824000;
}
fs = 78233600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 191*Wout*Hout] += -I[fs + 0]+I[fs + 1]-I[fs + 2]+I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 78233600;
}
fs = 78643200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 192*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]+I[fs + 640]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 78643200;
}
fs = 79052800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 193*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 79052800;
}
fs = 79462400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 194*Wout*Hout] += +I[fs + 1]+I[fs + 2]+I[fs + 641]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 79462400;
}
fs = 79872000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 195*Wout*Hout] += -I[fs + 1]+I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 79872000;
}
fs = 80281600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 196*Wout*Hout] += -I[fs + 2]+I[fs + 641]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 80281600;
}
fs = 80691200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 197*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 80691200;
}
fs = 81100800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 198*Wout*Hout] += +I[fs + 0]-I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 642]+0;
		fs += stride;
	}
	fs = stride*Win + 81100800;
}
fs = 81510400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 199*Wout*Hout] += -I[fs + 0]-I[fs + 642]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 81510400;
}
fs = 81920000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 200*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 81920000;
}
fs = 82329600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 201*Wout*Hout] += +I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 642]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 82329600;
}
fs = 82739200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 202*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 640]-I[fs + 641]+0;
		fs += stride;
	}
	fs = stride*Win + 82739200;
}
fs = 83148800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 203*Wout*Hout] += +I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 83148800;
}
fs = 83558400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 204*Wout*Hout] += +I[fs + 640]-I[fs + 641]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 83558400;
}
fs = 83968000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 205*Wout*Hout] += -I[fs + 0]-I[fs + 640]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 83968000;
}
fs = 84377600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 206*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 84377600;
}
fs = 84787200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 207*Wout*Hout] += -I[fs + 0]+I[fs + 2]-I[fs + 640]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 84787200;
}
fs = 85196800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 208*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 85196800;
}
fs = 85606400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 209*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 85606400;
}
fs = 86016000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 210*Wout*Hout] += -I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 86016000;
}
fs = 86425600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 211*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 640]+I[fs + 641]+I[fs + 642]+0;
		fs += stride;
	}
	fs = stride*Win + 86425600;
}
fs = 86835200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 212*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 640]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 86835200;
}
fs = 87244800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 213*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 87244800;
}
fs = 87654400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 214*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 87654400;
}
fs = 88064000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 215*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 641]-I[fs + 642]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 88064000;
}
fs = 88473600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 216*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 640]-I[fs + 641]+0;
		fs += stride;
	}
	fs = stride*Win + 88473600;
}
fs = 88883200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 217*Wout*Hout] += -I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 88883200;
}
fs = 89292800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 218*Wout*Hout] += +I[fs + 0]+I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 89292800;
}
fs = 89702400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 219*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 89702400;
}
fs = 90112000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 220*Wout*Hout] += +I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 90112000;
}
fs = 90521600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 221*Wout*Hout] += -I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 641]+I[fs + 642]+0;
		fs += stride;
	}
	fs = stride*Win + 90521600;
}
fs = 90931200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 222*Wout*Hout] += +I[fs + 0]+I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 90931200;
}
fs = 91340800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 223*Wout*Hout] += +I[fs + 1]-I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 642]-I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 91340800;
}
fs = 91750400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 224*Wout*Hout] += -I[fs + 0]+I[fs + 1]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 91750400;
}
fs = 92160000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 225*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 92160000;
}
fs = 92569600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 226*Wout*Hout] += -I[fs + 0]-I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 92569600;
}
fs = 92979200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 227*Wout*Hout] += +I[fs + 2]+I[fs + 640]+I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 92979200;
}
fs = 93388800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 228*Wout*Hout] += +I[fs + 0]-I[fs + 2]+I[fs + 641]+I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 93388800;
}
fs = 93798400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 229*Wout*Hout] += +I[fs + 0]+I[fs + 1]-I[fs + 2]-I[fs + 640]-I[fs + 641]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 93798400;
}
fs = 94208000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 230*Wout*Hout] += +I[fs + 0]-I[fs + 640]+I[fs + 641]+I[fs + 642]-I[fs + 1280]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 94208000;
}
fs = 94617600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 231*Wout*Hout] += +I[fs + 0]-I[fs + 640]+I[fs + 642]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 94617600;
}
fs = 95027200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 232*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 640]-I[fs + 641]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 95027200;
}
fs = 95436800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 233*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 640]+I[fs + 642]-I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 95436800;
}
fs = 95846400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 234*Wout*Hout] += +I[fs + 0]-I[fs + 641]+I[fs + 642]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 95846400;
}
fs = 96256000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 235*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 641]-I[fs + 642]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 96256000;
}
fs = 96665600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 236*Wout*Hout] += -I[fs + 1]-I[fs + 641]+I[fs + 642]+I[fs + 1280]-I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 96665600;
}
fs = 97075200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 237*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 97075200;
}
fs = 97484800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 238*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 640]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 97484800;
}
fs = 97894400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 239*Wout*Hout] += -I[fs + 1]-I[fs + 2]-I[fs + 640]+I[fs + 642]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 97894400;
}
fs = 98304000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 240*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 98304000;
}
fs = 98713600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 241*Wout*Hout] += +I[fs + 0]-I[fs + 1]+I[fs + 640]-I[fs + 641]+0;
		fs += stride;
	}
	fs = stride*Win + 98713600;
}
fs = 99123200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 242*Wout*Hout] += +I[fs + 0]-I[fs + 1]-I[fs + 2]+I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 99123200;
}
fs = 99532800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 243*Wout*Hout] += +I[fs + 0]-I[fs + 2]-I[fs + 640]+I[fs + 641]-I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 99532800;
}
fs = 99942400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 244*Wout*Hout] += -I[fs + 640]+I[fs + 641]-I[fs + 642]-I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 99942400;
}
fs = 100352000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 245*Wout*Hout] += -I[fs + 640]-I[fs + 641]+I[fs + 1280]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 100352000;
}
fs = 100761600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 246*Wout*Hout] += -I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 100761600;
}
fs = 101171200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 247*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]-I[fs + 642]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 101171200;
}
fs = 101580800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 248*Wout*Hout] += +I[fs + 0]+I[fs + 640]+I[fs + 641]-I[fs + 642]+0;
		fs += stride;
	}
	fs = stride*Win + 101580800;
}
fs = 101990400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 249*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 101990400;
}
fs = 102400000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 250*Wout*Hout] += +I[fs + 0]-I[fs + 640]+I[fs + 1280]+I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 102400000;
}
fs = 102809600;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 251*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 641]+I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 102809600;
}
fs = 103219200;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 252*Wout*Hout] += -I[fs + 0]-I[fs + 1]+I[fs + 2]-I[fs + 641]-I[fs + 1280]+0;
		fs += stride;
	}
	fs = stride*Win + 103219200;
}
fs = 103628800;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 253*Wout*Hout] += +I[fs + 0]+I[fs + 1]+I[fs + 2]+I[fs + 640]-I[fs + 641]-I[fs + 642]+I[fs + 1280]-I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 103628800;
}
fs = 104038400;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 254*Wout*Hout] += -I[fs + 1]+I[fs + 2]-I[fs + 640]+I[fs + 642]+I[fs + 1280]+I[fs + 1281]+0;
		fs += stride;
	}
	fs = stride*Win + 104038400;
}
fs = 104448000;
for (int i = 0; i < Hout; i++){
	for (int j = 0; j < Wout; j++){
		O[j + i*Wout + 255*Wout*Hout] += +I[fs + 0]+I[fs + 640]+I[fs + 641]-I[fs + 1280]-I[fs + 1281]+I[fs + 1282]+0;
		fs += stride;
	}
	fs = stride*Win + 104448000;
}

}

void conv2d_depth_v3(int I[], int O[]){
    memset(O, 0, OUTARRLEN*sizeof(int));
    long int fs = 0;
    /*V3*/
}

void conv2d_depth_v4(int I[], int O[]){
    memset(O, 0, OUTARRLEN*sizeof(int));
    long int fs = 0;
    /*V4*/
}

void conv2d_depth_naive(int I[], int O[], int W[], int depth, int kernel_size_, int stride_){
    memset(O, 0, OUTARRLEN*sizeof(int));
    long int fs = 0;
    for (int i = 0; i < Hout; i++){
        for (int j = 0; j < Wout; j++){
            for (int d = 0; d < depth; d++){
                for (int h = 0; h < kernel_size_; h++){
                    for (int w = 0; w < kernel_size_; w++){
                        O[j + i*Wout + d*Wout*Hout] += W[w + h*kernel_size_ + d*kernel_size_*kernel_size_] * I[fs + w + h*Win + d*Win*Hin];
                    }
                }
            }
            fs += stride_;
        }
        fs = Win*stride_;
    }

}

void conv2d_depth_naive_inverted(int I[], int O[], int W[], int depth, int kernel_size_, int stride_){
    memset(O, 0, OUTARRLEN*sizeof(int));
    long int fs = 0;
    for (int d = 0; d < depth; d++){
        fs = 0;
        for (int i = 0; i < Hout; i++){
            for (int j = 0; j < Hout; j++){
                for (int h = 0; h < kernel_size_; h++){
                    for (int w = 0; w < kernel_size_; w++){
                        O[j + i*Wout + d*Wout*Hout] += W[w + h*kernel_size_ + d*kernel_size_*kernel_size_] * I[fs + w + h*Win + d*Win*Hin];
                    }
                }
            }
            fs += stride_;
        }
        fs = Win*stride_;
    }
}

void conv2d_depth_stride_of_one(int I[], int O[], int W[], int depth, int kernel_size_){
    memset(O, 0, OUTARRLEN*sizeof(int));
    for (int d = 0; d < depth; d++){
        for (int h = 0; h < kernel_size_; h++){
            for (int w = 0; w < kernel_size_; w++){

                if (W[w + h*kernel_size_ + d*kernel_size_*kernel_size] == 1){
                    long int fs = w + h*Win + d*Win*Hin;
                    for (long int ac0 = h; ac0 < Hout + h; ac0++){
                        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
                            O[ac1 - w - h*Wout] += I[ac1];
                        }
                        fs += Win;
                    }
                }

                if (W[w + h*kernel_size_ + d*kernel_size_*kernel_size] == -1){
                    long int fs = w + h*Win + d*Win*Hin;
                    for (long int ac0 = h; ac0 < Hout + h; ac0++){
                        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
                            O[ac1 - w - h*Wout] -= I[ac1];
                        }
                        fs += Win;
                    }
                }

            }
        }
    }
}


void f000(int I[], int O[], int d, int h){//-1-1-1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1] - I[ac1+1] - I[ac1+2];
        }
        fs += Win;
    }
}

void f001(int I[], int O[], int d, int h){//-1-1 0
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1] - I[ac1+1];
        }
        fs += Win;
    }
}

void f002(int I[], int O[], int d, int h){//-1 -1 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1] - I[ac1+1] + I[ac1+2];
        }
        fs += Win;
    }
}

void f010(int I[], int O[], int d, int h){//1 0 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1]  + I[ac1+2];
        }
        fs += Win;
    }
}

void f011(int I[], int O[], int d, int h){//-1 0 0 
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1];
        }
        fs += Win;
    }
}

void f012(int I[], int O[], int d, int h){//-1 0 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1] + I[ac1+2];
        }
        fs += Win;
    }
}

void f020(int I[], int O[], int d, int h){//-1 1 -1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1] + I[ac1+1] - I[ac1+2];
        }
        fs += Win;
    }
}

void f021(int I[], int O[], int d, int h){//-1 1 0
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1] + I[ac1+1];
        }
        fs += Win;
    }
}

void f022(int I[], int O[], int d, int h){//-1 1 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += -I[ac1] + I[ac1+1] + I[ac1+2];
        }
        fs += Win;
    }
}

void f100(int I[], int O[], int d, int h){//0 -1 -1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] +=  - I[ac1+1] - I[ac1+2];
        }
        fs += Win;
    }
}

void f101(int I[], int O[], int d, int h){//0 -1 0
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += - I[ac1+1];
        }
        fs += Win;
    }
}

void f102(int I[], int O[], int d, int h){//0 -1 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += - I[ac1+1] + I[ac1+2];
        }
        fs += Win;
    }
}

void f110(int I[], int O[], int d, int h){//0 0 -1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] +=  - I[ac1+2];
        }
        fs += Win;
    }
}

void f111(int I[], int O[], int d, int h){//0 0 0

}

void f112(int I[], int O[], int d, int h){//0 0 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] +=  + I[ac1+2];
        }
        fs += Win;
    }
}

void f120(int I[], int O[], int d, int h){//0 1 -1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += + I[ac1+1] - I[ac1+2];
        }
        fs += Win;
    }
}

void f121(int I[], int O[], int d, int h){//0 1 0
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += + I[ac1+1];
        }
        fs += Win;
    }
}

void f122(int I[], int O[], int d, int h){//0 1 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += + I[ac1+1] + I[ac1+2];
        }
        fs += Win;
    }
}

void f200(int I[], int O[], int d, int h){//1 -1 -1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1] - I[ac1+1] - I[ac1+2];
        }
        fs += Win;
    }
}

void f201(int I[], int O[], int d, int h){//1 -1 0
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1] - I[ac1+1];
        }
        fs += Win;
    }
}

void f202(int I[], int O[], int d, int h){//1 -1 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1] - I[ac1+1] + I[ac1+2];
        }
        fs += Win;
    }
}

void f210(int I[], int O[], int d, int h){//1 0 -1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1] - I[ac1+2];
        }
        fs += Win;
    }
}

void f211(int I[], int O[], int d, int h){//1 0 0
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1];
        }
        fs += Win;
    }
}

void f212(int I[], int O[], int d, int h){//1 0 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1]  + I[ac1+2];
        }
        fs += Win;
    }
}

void f220(int I[], int O[], int d, int h){//1 1 -1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1] + I[ac1+1] - I[ac1+2];
        }
        fs += Win;
    }
}

void f221(int I[], int O[], int d, int h){//1 1 0
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1] + I[ac1+1];
        }
        fs += Win;
    }
}

void f222(int I[], int O[], int d, int h){//1 1 1
    long int fs = h*Win + d*Win*Hin;
    for (long int ac0 = h; ac0 < Hout + h; ac0++){
        for (long int ac1 = fs; ac1 < Wout + fs; ac1++){
            O[ac1 - h*Wout] += +I[ac1] + I[ac1+1] + I[ac1+2];
        }
        fs += Win;
    }
}

void conv2d_depth_piecewise_stride_one(int I[], int O[], int W[], int depth, int kernel_size_){
    for (int d = 0; d < depth; d++){
        for (int h = 0; h < kernel_size_; h++){
            void (*f_to_do)(int*, int*, int, int);
            int piece = 100*(W[0+h*kernel_size_+d*kernel_size_*kernel_size_]+1) + 10*(W[1+h*kernel_size_+d*kernel_size_*kernel_size_]+1) + 1*(W[2+h*kernel_size_+d*kernel_size_*kernel_size_]+1);
            switch(piece){
                case 0:
                    f_to_do = f000;
                    break;
                case 1:
                    f_to_do = f001;
                    break;
                case 2:
                    f_to_do = f002;
                    break;
                case 10:
                    f_to_do = f010;
                    break;
                case 11:
                    f_to_do = f011;
                    break;
                case 12:
                    f_to_do = f012;
                    break;
                case 20:
                    f_to_do = f020;
                    break;
                case 21:
                    f_to_do = f021;
                    break;
                case 22:
                    f_to_do = f022;
                    break;

                case 100:
                    f_to_do = f100;
                    break;
                case 101:
                    f_to_do = f101;
                    break;
                case 102:
                    f_to_do = f102;
                    break;
                case 110:
                    f_to_do = f110;
                    break;
                case 111:
                    f_to_do = f111;
                    break;
                case 112:
                    f_to_do = f112;
                    break;
                case 120:
                    f_to_do = f120;
                    break;
                case 121:
                    f_to_do = f121;
                    break;
                case 122:
                    f_to_do = f122;
                    break;

                case 200:
                    f_to_do = f200;
                    break;
                case 201:
                    f_to_do = f201;
                    break;
                case 202:
                    f_to_do = f202;
                    break;
                case 210:
                    f_to_do = f210;
                    break;
                case 211:
                    f_to_do = f211;
                    break;
                case 212:
                    f_to_do = f212;
                    break;
                case 220:
                    f_to_do = f220;
                    break;
                case 221:
                    f_to_do = f221;
                    break;
                case 222:
                    f_to_do = f222;
                    break;              
            }
            f_to_do(I, O, d, h);
        }
    }
}



void conv2d_depth_new(int I[], int O[]){

}

int main(){
    for (long int i = 0; i < INARRLEN; i++){A[i] = rand()%512 - 256;}
    FILE* wfile = fopen(wfilename, "r");
    for (int i = 0; i < Din*kernel_size*kernel_size; i++){fscanf(wfile, "%d\n", &W[i]);}
    fclose(wfile);
    int max;

    clock_t time_v2_beg = clock();
    conv2d_depth_v2(A, B);
    clock_t time_v2_end = clock();
    printf("time for conv2d depth v2: %f s\n", (double)(time_v2_end - time_v2_beg)/CLOCKS_PER_SEC);
    max = B[0];for (int i = 1; i < OUTARRLEN; i++){if (B[i] > max){max = B[i];}}
    FILE* garbagev2 = fopen("garbage2.gbg", "w");fprintf(garbagev2, "%d", max);fclose(garbagev2);



    clock_t time_vn_beg = clock();
    conv2d_depth_naive(A, B, W, Din, kernel_size, stride);
    clock_t time_vn_end = clock();
    printf("time for conv2d depth version naive: %f s\n", (double)(time_vn_end - time_vn_beg)/CLOCKS_PER_SEC);
    max = B[0];for (int i = 1; i < OUTARRLEN; i++){if (B[i] > max){max = B[i];}}
    FILE* garbagevn = fopen("garbagen.gbg", "w");fprintf(garbagevn, "%d", max);fclose(garbagevn);

    clock_t time_vni_beg = clock();
    conv2d_depth_naive_inverted(A, B, W, Din, kernel_size, stride);
    clock_t time_vni_end = clock();
    printf("time for conv2d depth version naive-inverted: %f s\n", (double)(time_vni_end - time_vni_beg)/CLOCKS_PER_SEC);
    max = B[0];for (int i = 1; i < OUTARRLEN; i++){if (B[i] > max){max = B[i];}}
    FILE* garbagevni = fopen("garbageni.gbg", "w");fprintf(garbagevni, "%d", max);fclose(garbagevni);

    clock_t time_vbeloved_beg = clock();
    conv2d_depth_stride_of_one(A, B, W, Din, kernel_size);
    clock_t time_vbeloved_end = clock();
    printf("time for conv2d depth version assuming stride of 1: %f s\n", (double)(time_vbeloved_end - time_vbeloved_beg)/CLOCKS_PER_SEC);
    max = B[0];for (int i = 1; i < OUTARRLEN; i++){if (B[i] > max){max = B[i];}}
    FILE* garbagevbelovedi = fopen("garbagevbeloved.gbg", "w");fprintf(garbagevbelovedi, "%d", max);fclose(garbagevbelovedi);

    clock_t time_vbelovedp_beg = clock();
    conv2d_depth_piecewise_stride_one(A, B, W, Din, kernel_size);
    clock_t time_vbelovedp_end = clock();
    printf("time for conv2d depth version assuming stride of 1 and also piecewise: %f s\n", (double)(time_vbelovedp_end - time_vbelovedp_beg)/CLOCKS_PER_SEC);
    max = B[0];for (int i = 1; i < OUTARRLEN; i++){if (B[i] > max){max = B[i];}}
    FILE* garbagevbelovedip = fopen("garbagevbelovedp.gbg", "w");fprintf(garbagevbelovedip, "%d", max);fclose(garbagevbelovedip);

    return 0;
}

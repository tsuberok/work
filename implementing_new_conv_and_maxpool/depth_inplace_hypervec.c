#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#define DCONST0 8

void conv_depth_transposed_inplace_hypervectorized(int I[], int W[], int D, int Hin, int Win, int kernel_size, int stride, int padding, int thread_no, int NUMTHREADS, uint16_t isCoal[]){
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

     int*  scratchpad_output = (int*)malloc(memamc*D*Wout*sizeof(int));
    memory_bank mem_bank;
    mem_bank.max_size = memamc;
    mem_bank.size_free = 0;
    mem_bank.size_occupied = 0;
    mem_bank.free_cells = (int*)malloc(mem_bank.max_size*sizeof(int));
    mem_bank.occupied_cells = (int*)malloc(mem_bank.max_size*sizeof(int));
    for (int i = 0; i < mem_bank.max_size; i++){
        //printf("I AM PUSHING %d TO MEM BANK FREE CELLS\n", i);
        mb_push(i, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
    }


     __m256i filter00[DCONST0];
     __m256i filter01[DCONST0];
     __m256i filter02[DCONST0];
     __m256i filter10[DCONST0];
     __m256i filter11[DCONST0];
     __m256i filter12[DCONST0];
     __m256i filter20[DCONST0];
     __m256i filter21[DCONST0];
     __m256i filter22[DCONST0];
    for (int n = 0; n < DCONST0; n++){filter00[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*0]);}
    for (int n = 0; n < DCONST0; n++){filter01[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*1]);}
    for (int n = 0; n < DCONST0; n++){filter02[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*2]);}
    for (int n = 0; n < DCONST0; n++){filter10[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*3]);}
    for (int n = 0; n < DCONST0; n++){filter11[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*4]);}
    for (int n = 0; n < DCONST0; n++){filter12[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*5]);}
    for (int n = 0; n < DCONST0; n++){filter20[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*6]);}
    for (int n = 0; n < DCONST0; n++){filter21[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*7]);}
    for (int n = 0; n < DCONST0; n++){filter22[n] = _mm256_loadu_si256((__m256i*)&W[n*8 + D*8]);}
    //printf("thread_no = %d. LOADED THE FILTER\n", thread_no);
     __m256i cur00[DCONST0];
     __m256i cur01[DCONST0];
     __m256i cur02[DCONST0];
     __m256i cur10[DCONST0];
     __m256i cur11[DCONST0];
     __m256i cur12[DCONST0];
     __m256i cur20[DCONST0];
     __m256i cur21[DCONST0];
     __m256i cur22[DCONST0];

     __m256i acc[DCONST0];
    int arrzeros[8];


    int iin = iin_from_iout(ioutstart, stride, padding);
    int outOfBounds;
    int num_to_do;

    for (int i = ioutstart; i < ioutend; i++){
        //printf("thread_no = %d i = %d iin = %d\n",thread_no, i, iin);
        int poss = mb_pop(mem_bank.free_cells, &mem_bank.size_free);
        //printf("thread_no %d. popped from free cells and poss = %d. mem_bank.size_free = %d\n", thread_no, poss, mem_bank.size_free);
        mb_push(poss, mem_bank.occupied_cells, &mem_bank.size_occupied, mem_bank.max_size);
        memset(&scratchpad_output[poss*D*Wout], 0, D*Wout*sizeof(int));

        for (int j = 0; j < Wout; j++){
            for (int n = 0; n < DCONST0; n++){acc[n] = _mm256_loadu_si256((__m256i*)arrzeros);}

            if (iin > -1){
                if (j*stride-padding >= 0){
                    for ( int n = 0; n < DCONST0; n++){
                        int isConvDone = 0;
                        if (isHeader && iin+0 >= header_start_i && iin+0 < header_start_i + headeramc){
                            //conv1d(&header[D*Win*(iin+h - header_start_i) + start_in_iin*D], &scratchpad_output[poss*D*Wout + start_iout*D], cur_subfilter, num_to_do, D, stride, thread_no);
                            cur00[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 0 - header_start_i)*Win*D + n*8 + D*0 + (j*stride-padding)*D ]);
                            isConvDone = 1;
                        }
                        if (isFooter && iin+0 >= footer_start_i && iin+0 < footer_start_i + footeramc){
                            //conv1d(&footer[D*Win*(iin+h - footer_start_i) + start_in_iin*D], &scratchpad_output[poss*D*Wout + start_iout*D], cur_subfilter, num_to_do, D, stride, thread_no);
                            cur00[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 0 - footer_start_i)*Win*D +  n*8 + D*0 + (j*stride-padding)*D ]);
                            isConvDone = 1;
                        }
                        if (!isConvDone){
                            //printf("trying to load regular cur00\n");
                            cur00[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 0)*Win*D + n*8 + D*0 + (j*stride-padding)*D ]);
                            //printf("regular loading success cur00\n");
                        }

                        //printf("thread_no = %d. Loaded cur00 at n = %d\n", thread_no, n);
                        cur00[n] = _mm256_mullo_epi32(cur00[n], filter00[n]);
                        acc[n] = _mm256_add_epi32(acc[n], cur00[n]);
                    }
                }
                for ( int n = 0; n < DCONST0; n++){
                    int isConvDone = 0;
                    if (isHeader && iin+0 >= header_start_i && iin+0 < header_start_i + headeramc){
                        cur01[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 0 - header_start_i)*Win*D + n*8 + D*1 + (j*stride-padding)*D ]);
                        isConvDone = 1;
                    }
                    if (isFooter && iin+0 >= footer_start_i && iin+0 < footer_start_i + footeramc){
                        cur01[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 0 - footer_start_i)*Win*D +  n*8 + D*1 + (j*stride-padding)*D ]);
                        isConvDone = 1;    
                    }
                    if (!isConvDone){
                        //printf("trying to load regular cur01\n");
                        cur01[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 0)*Win*D + n*8 + D*1 + (j*stride-padding)*D ]);
                        //printf("regular loading success cur01\n");
                    }
                    //printf("thread_no = %d. Loaded cur01 at n = %d\n", thread_no, n);

                    cur01[n] = _mm256_mullo_epi32(cur01[n], filter01[n]);
                    acc[n] = _mm256_add_epi32(acc[n], cur01[n]);
                }
                if (j*stride-padding + 2 < Win){
                    for ( int n = 0; n < DCONST0; n++){
                        int isConvDone = 0;
                        if (isHeader && iin+0 >= header_start_i && iin+0 < header_start_i + headeramc){
                            cur02[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 0 - header_start_i)*Win*D + n*8 + D*2 + (j*stride-padding)*D ]);
                            isConvDone = 1;
                        }
                        if (isFooter && iin+0 >= footer_start_i && iin+0 < footer_start_i + footeramc){
                            cur02[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 0 - footer_start_i)*Win*D +  n*8 + D*2 + (j*stride-padding)*D ]);
                            isConvDone = 1;   
                        }
                        if (!isConvDone){
                            //printf("trying to load regular cur02\n");
                            cur02[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 0)*Win*D + n*8 + D*2 + (j*stride-padding)*D ]);
                            //printf("regular loading success cur02\n");
                        }
                        //printf("thread_no = %d. Loaded cur02 at n = %d\n", thread_no, n);
                        cur02[n] = _mm256_mullo_epi32(cur02[n], filter02[n]);
                        acc[n] = _mm256_add_epi32(acc[n], cur02[n]);
                    }
                }
            }

            if (j*stride-padding >= 0){
                for ( int n = 0; n < DCONST0; n++){
                    int isConvDone = 0;
                    if (isHeader && iin+1 >= header_start_i && iin+1 < header_start_i + headeramc){
                        cur10[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 1 - header_start_i)*Win*D + n*8 + D*0 + (j*stride-padding)*D ]);
                        isConvDone = 1;
                    }
                    if (isFooter && iin+1 >= footer_start_i && iin+1 < footer_start_i + footeramc){
                        cur10[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 1 - footer_start_i)*Win*D +  n*8 + D*0 + (j*stride-padding)*D ]);
                        isConvDone = 1;  
                    }
                    if (!isConvDone){
                        cur10[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 1)*Win*D + n*8 + D*0 + (j*stride-padding)*D ]);
                    }
                    //printf("thread_no = %d. Loaded cur10 at n = %d\n", thread_no, n);
                    cur10[n] = _mm256_mullo_epi32(cur10[n], filter10[n]);
                    acc[n] = _mm256_add_epi32(acc[n], cur10[n]);
                }
            }
            for ( int n = 0; n < DCONST0; n++){
                int isConvDone = 0;
                if (isHeader && iin+1 >= header_start_i && iin+1 < header_start_i + headeramc){
                    cur11[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 1 - header_start_i)*Win*D + n*8 + D*1 + (j*stride-padding)*D ]);
                    isConvDone = 1;
                }
                if (isFooter && iin+1 >= footer_start_i && iin+1 < footer_start_i + footeramc){
                    cur11[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 1 - footer_start_i)*Win*D +  n*8 + D*1 + (j*stride-padding)*D ]);
                    isConvDone = 1;  
                }
                if (!isConvDone){
                    cur11[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 1)*Win*D + n*8 + D*1 + (j*stride-padding)*D ]);
                }
                //printf("thread_no = %d. Loaded cur11 at n = %d\n", thread_no, n);
                cur11[n] = _mm256_mullo_epi32(cur11[n], filter11[n]);
                acc[n] = _mm256_add_epi32(acc[n], cur11[n]);
            }
            if (j*stride-padding + 2 < Win){
                for ( int n = 0; n < DCONST0; n++){
                    int isConvDone = 0;
                    if (isHeader && iin+1 >= header_start_i && iin+1 < header_start_i + headeramc){
                        cur12[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 1 - header_start_i)*Win*D + n*8 + D*2 + (j*stride-padding)*D ]);
                        isConvDone = 1;
                    }
                    if (isFooter && iin+1 >= footer_start_i && iin+1 < footer_start_i + footeramc){
                        cur12[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 1 - footer_start_i)*Win*D +  n*8 + D*2 + (j*stride-padding)*D ]);
                        isConvDone = 1;  
                    }
                    if (!isConvDone){
                        cur12[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 1)*Win*D + n*8 + D*2 + (j*stride-padding)*D ]);
                    }
                    //printf("thread_no = %d. Loaded cur12 at n = %d\n", thread_no, n);
                    cur12[n] = _mm256_mullo_epi32(cur12[n], filter12[n]);
                    acc[n] = _mm256_add_epi32(acc[n], cur12[n]);
                }
            }

            if (iin + 2 < Hin){
                if (j*stride-padding  >= 0){
                    for ( int n = 0; n < DCONST0; n++){
                        int isConvDone = 0;
                        if (isHeader && iin+2 >= header_start_i && iin+2 < header_start_i + headeramc){
                            cur20[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 2 - header_start_i)*Win*D + n*8 + D*0 + (j*stride-padding)*D ]);
                            isConvDone = 1;
                        }
                        if (isFooter && iin+2 >= footer_start_i && iin+2 < footer_start_i + footeramc){
                            cur20[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 2 - footer_start_i)*Win*D +  n*8 + D*0 + (j*stride-padding)*D ]);
                            isConvDone = 1;  
                        }
                        if (!isConvDone){
                            cur20[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 2)*Win*D + n*8 + D*0 + (j*stride-padding)*D ]);
                        }
                        //printf("thread_no = %d. Loaded cur20 at n = %d\n", thread_no, n);
                        cur20[n] = _mm256_mullo_epi32(cur20[n], filter20[n]);
                        acc[n] = _mm256_add_epi32(acc[n], cur20[n]);
                    }
                }
                for ( int n = 0; n < DCONST0; n++){
                    int isConvDone = 0;
                    if (isHeader && iin+2 >= header_start_i && iin+2 < header_start_i + headeramc){
                        cur21[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 2 - header_start_i)*Win*D + n*8 + D*1 + (j*stride-padding)*D ]);
                        isConvDone = 1;
                    }
                    if (isFooter && iin+2 >= footer_start_i && iin+2 < footer_start_i + footeramc){
                        cur21[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 2 - footer_start_i)*Win*D +  n*8 + D*1 + (j*stride-padding)*D ]);
                        isConvDone = 1;  
                    }
                    if (!isConvDone){
                        cur21[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 2)*Win*D + n*8 + D*1 + (j*stride-padding)*D ]);
                    }
                    //printf("thread_no = %d. Loaded cur21 at n = %d\n", thread_no, n);
                    cur21[n] = _mm256_mullo_epi32(cur21[n], filter21[n]);
                    acc[n] = _mm256_add_epi32(acc[n], cur21[n]);
                }
                if (j*stride-padding  + 2 < Win){
                    for ( int n = 0; n < DCONST0; n++){
                        int isConvDone = 0;
                        if (isHeader && iin+2 >= header_start_i && iin+2 < header_start_i + headeramc){
                            cur22[n] = _mm256_loadu_si256((__m256i*)&header[(iin + 2 - header_start_i)*Win*D + n*8 + D*2 + (j*stride-padding)*D ]);
                            isConvDone = 1;
                        }
                        if (isFooter && iin+2 >= footer_start_i && iin+2 < footer_start_i + footeramc){
                            cur22[n] = _mm256_loadu_si256((__m256i*)&footer[(iin  + 2 - footer_start_i)*Win*D +  n*8 + D*2 + (j*stride-padding)*D ]);
                            isConvDone = 1;  
                        }
                        if (!isConvDone){
                            cur22[n] = _mm256_loadu_si256((__m256i*)&I[(iin  + 2)*Win*D + n*8 + D*2 + (j*stride-padding)*D ]);
                        }
                        //printf("thread_no = %d. Loaded cur22 at n = %d\n", thread_no, n);
                        cur22[n] = _mm256_mullo_epi32(cur22[n], filter22[n]);
                        acc[n] = _mm256_add_epi32(acc[n], cur22[n]);
                    }
                }
            }

            for ( int n = 0; n < DCONST0; n++){
                _mm256_storeu_si256((__m256i*)&scratchpad_output[poss*Wout*D + j*D+ n*8] , acc[n]);
            }


        }

        if (mem_bank.size_occupied == mem_bank.max_size){
            //printf("during attemt at writing, mem_bank.size_occupied = %d and the full arr is: ", mem_bank.size_occupied);
            //for (int suka = 0; suka < mem_bank.max_size; suka++){printf("%d ", mem_bank.occupied_cells[suka]);}printf("\n");
            int poss_output = mb_pop(mem_bank.occupied_cells, &mem_bank.size_occupied);
            int stride_factor = 0;
            if (stride > 1){
                stride_factor = padding;
            }
            //printf("thread_no = %d. WRITE from poss_output = %d to iin+srtride factor = %d. Copying %d ints. From address = %p to adress %p\n", thread_no, poss_output, iin+stride_factor, D*Wout, &scratchpad_output[poss_output*D*Wout], rows_to_process[iin]);

            memcpy(&I[(iin+stride_factor)*D*Win], &scratchpad_output[poss_output*D*Wout], D*Wout*sizeof(int));

            mb_push(poss_output, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
        }
        iin += stride;  
        //printf("reached the uhh the end of i one cycle\n");
    }
    while(mem_bank.size_occupied > 0){
        int poss_output = mb_pop(mem_bank.occupied_cells, &mem_bank.size_occupied);
        int stride_factor = 0;
        if (stride > 1){
            stride_factor = padding;
        }
        //printf("thread_no = %d. FINAL WRITE from poss_output = %d to iin + stride_factor = %d. Copying %d ints. From address = %p to adress %p\n", thread_no, poss_output, iin+stride_factor, D*Wout, &scratchpad_output[poss_output*D*Wout], rows_to_process[iin]);
        //printf("scratchpad output is:\n");
        //for(int suka = poss_output*D*Wout; suka < poss_output*D*Wout + D*Wout; suka++){printf("%d\n", scratchpad_output[suka]);}

        memcpy(&I[(iin+stride_factor)*D*Win], &scratchpad_output[poss_output*D*Wout], D*Wout*sizeof(int));

        mb_push(poss_output, mem_bank.free_cells, &mem_bank.size_free, mem_bank.max_size);
        iin += stride;
    }
    free(scratchpad_output);
    free(mem_bank.free_cells);
    free(mem_bank.occupied_cells);

    if (isHeader){free(header);}
    if (isFooter){free(footer);}
    //printf("thread_no = %d. Got so far\n", thread_no);
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
        //printf("thread_no = %d. ATTEMPTING COAL\n", thread_no);
        coal_mem(I, segments, finals, seg_lens, isCoal, Hout,  thread_no,  NUMTHREADS);
        free(segments);
        free(seg_lens);
        free(finals);
    }
}
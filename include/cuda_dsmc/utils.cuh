/** \file
 *  \brief Vector functions
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef UTILS_CUH_INCLUDED
#define UTILS_CUH_INCLUDED 1

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CALL(x) do {                            \
    if((x)!=cudaSuccess) {                           \
        printf("Error at %s:%d\n",__FILE__,__LINE__);\
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

#define CURAND_CALL(x) do {                          \
    if((x)!=CURAND_STATUS_SUCCESS) {                 \
        printf("Error at %s:%d\n",__FILE__,__LINE__);\
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

#endif  // UTILS_CUH_INCLUDED
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

void numberElementsPerParallelUnit(int unit_id,
                                   int num_units,
                                   int *num_elements);

void combineDeviceArrays(int num_devices,
                         int num_elements,
                         double3** device_arrays,
                         double3* host_array);

#endif  // UTILS_CUH_INCLUDED
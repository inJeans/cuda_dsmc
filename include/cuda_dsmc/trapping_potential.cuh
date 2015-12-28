/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef TRAPPING_POTENTIAL_CUH_INCLUDED
#define TRAPPING_POTENTIAL_CUH_INCLUDED 1

// #ifdef CUDA
#include <cuda_runtime.h>
// #endif

typedef struct trap_geo{
    double Bz;
    double B0;
} trap_geo;

 __host__ __device__ double3 B(double3 r,
                               trap_geo params);

#endif  // TRAPPING_POTENTIAL_CUH_INCLUDED
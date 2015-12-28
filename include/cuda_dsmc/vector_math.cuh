/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef VECTOR_MATH_CUH_INCLUDED
#define VECTOR_MATH_CUH_INCLUDED 1

// #ifdef CUDA
#include <cuda_runtime.h>
// #endif

#include <math.h>

static __inline__ __host__ __device__ double3 operator*(double3 a, 
                                                        double b) {
    return make_double3(a.x*b, a.y*b, a.z*b);
}

static __inline__ __host__ __device__ double3 operator/(double3 a, 
                                                        double b) {
    return make_double3(a.x/b, a.y/b, a.z/b);
}

__host__ __device__ double norm(double3 vec);

#endif  // VECTOR_MATH_CUH_INCLUDED
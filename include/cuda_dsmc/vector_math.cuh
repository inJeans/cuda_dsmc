/** \file
 *  \brief Vector functions
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef VECTOR_MATH_CUH_INCLUDED
#define VECTOR_MATH_CUH_INCLUDED 1

#include <cuda_runtime.h>
#include <math.h>

static __inline__ __host__ __device__ double2 operator/(double2 a, 
                                                        double b) {
    return make_double2(a.x/b, a.y/b);
}

static __inline__ __host__ __device__ double2 operator-(double2 a, 
                                                        double b) {
    return make_double2(a.x-b, a.y-b);
}

#endif  // VECTOR_MATH_CUH_INCLUDED
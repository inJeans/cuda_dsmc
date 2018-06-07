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

///////////////
//  DOUBLE3  //
///////////////

static __inline__ __host__ __device__ double norm(double3 a) {
    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

static __inline__ __host__ __device__ double3 operator*(double3 a, 
                                                        double3 b) {
    return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

static __inline__ __host__ __device__ double3 operator*(double a, 
                                                        double3 b) {
    return make_double3(a*b.x, a*b.y, a*b.z);
}

static __inline__ __host__ __device__ double3 operator/(double3 a, 
                                                        double b) {
    return make_double3(a.x/b, a.y/b, a.z/b);
}

static __inline__ __host__ __device__ double3 operator+(double3 a, 
                                                        double3 b) {
    return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

static __inline__ __host__ __device__ double3 operator+(double a, 
                                                        double3 b) {
    return make_double3(a+b.x, a+b.y, a+b.z);
}

static __inline__ __host__ __device__ double3 operator-(double3 a, 
                                                        double3 b) {
    return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}

static __inline__ __host__ __device__ double3 operator-(double3 a, 
                                                        double b) {
    return make_double3(a.x-b, a.y-b, a.z-b);
}

///////////////
//  DOUBLE2  //
///////////////

static __inline__ __host__ __device__ double2 operator/(double2 a, 
                                                        double b) {
    return make_double2(a.x/b, a.y/b);
}

static __inline__ __host__ __device__ double2 operator-(double2 a, 
                                                        double b) {
    return make_double2(a.x-b, a.y-b);
}

#endif  // VECTOR_MATH_CUH_INCLUDED
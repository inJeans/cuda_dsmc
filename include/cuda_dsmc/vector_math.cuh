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

#include "cuComplex.h"
struct zomplex2 {
    cuDoubleComplex up, dn;
};
typedef struct zomplex2 zomplex2;

static __inline__ __host__ __device__ zomplex2 make_zomplex2(double x, double y, double z, double w) {
  zomplex2 t; t.up.x = x; t.up.y = y; t.dn.x = z; t.dn.y = w; return t;
}

struct wavefunction {
    cuDoubleComplex up, dn;
    bool isSpinUp = TRUE;
};
typedef struct wavefunction wavefunction;

static __inline__ __host__ __device__ wavefunction make_wavefunction(double x, double y, double z, double w, bool isSpinUp) {
  wavefunction t; 
  
  t.up.x = x; t.up.y = y; t.dn.x = z; t.dn.y = w; 
  t.isSpinUp = isSpinUp;

  return t;
}

static __inline__ __host__ __device__ double3 operator*(double3 a, 
                                                        double b) {
    return make_double3(a.x*b, a.y*b, a.z*b);
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
    return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

static __inline__ __host__ __device__ double3 operator+(double a, 
                                                        double3 b) {
    return make_double3(a+b.x, a+b.y, a+b.z);
}

static __inline__ __host__ __device__ double3 operator+(double3 a, 
                                                        double b) {
    return make_double3(a.x+b, a.y+b, a.z+b);
}

static __inline__ __host__ __device__ cuDoubleComplex operator*(cuDoubleComplex a, 
                                                                cuDoubleComplex b) {
    return cuCmul(a, b);
}

static __inline__ __host__ __device__ cuDoubleComplex operator*(double a, 
                                                                cuDoubleComplex b) {
    return make_cuDoubleComplex(a*b.x, a*b.y);
}

static __inline__ __host__ __device__ cuDoubleComplex operator/(cuDoubleComplex a, 
                                                                int b) {
    return make_cuDoubleComplex(a.x/b, a.y/b);
}

static __inline__ __host__ __device__ cuDoubleComplex operator+(cuDoubleComplex a, 
                                                                cuDoubleComplex b) {
    return cuCadd(a, b);
}

// static __inline__ __host__ __device__ cuDoubleComplex& operator+=(const cuDoubleComplex &a, 
//                                                                   const cuDoubleComplex &b) {
//     cuDoubleComplex output = cuCadd(a, b);
//     return &output;
// }

static __inline__ __host__ __device__ cuDoubleComplex operator+(double a, 
                                                                cuDoubleComplex b) {
    return make_cuDoubleComplex(a+b.x, b.y);
}

static __inline__ __host__ __device__ cuDoubleComplex operator+(cuDoubleComplex a, 
                                                                double b) {
    return make_cuDoubleComplex(a.x+b, a.y);
}


static __inline__ __host__ __device__ cuDoubleComplex operator-(cuDoubleComplex a, 
                                                                double b) {
    return make_cuDoubleComplex(a.x-b, a.y);
}

__host__ __device__ double dot(double3 a, double3 b);

__host__ __device__ double3 unit(double3 vec);

__host__ __device__ double norm(double3 vec);

#endif  // VECTOR_MATH_CUH_INCLUDED
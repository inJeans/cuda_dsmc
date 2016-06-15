/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef TRAPPING_POTENTIAL_CUH_INCLUDED
#define TRAPPING_POTENTIAL_CUH_INCLUDED 1

// #ifdef CUDA
#include <cuda_runtime.h>
// #endif
#include "vector_math.cuh"

#if defined(IOFFE) // Ioffe Pritchard trap
typedef struct trap_geo{
    double B0;
    double dB;
    double ddB;
} trap_geo;
#else // Quadrupole trap
typedef struct trap_geo{
    double Bz;
    double B0;
} trap_geo;
#endif

__host__ __device__ double3 B(double3 r,
                              trap_geo params);

__host__ __device__ double3 dB_dx(double3 pos,
                                  trap_geo params);

__host__ __device__ double3 dB_dy(double3 pos,
                                  trap_geo params);

__host__ __device__ double3 dB_dz(double3 pos,
                                  trap_geo params);

__device__ double d_dV_dx(double3 pos,
                          trap_geo params);

__device__ double d_expectation_dV_dx(trap_geo params,
                                      double3 pos,
                                      zomplex2 psi);

__device__ double d_dV_dy(double3 pos,
                          trap_geo params);

__device__ double d_expectation_dV_dy(trap_geo params,
                                      double3 pos,
                                      zomplex2 psi);

__device__ double d_dV_dz(double3 pos,
                          trap_geo params);

__device__ double d_expectation_dV_dz(trap_geo params,
                                      double3 pos,
                                      zomplex2 psi);

#endif  // TRAPPING_POTENTIAL_CUH_INCLUDED
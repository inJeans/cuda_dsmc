/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef TEST_HELPERS_CUH_INCLUDED
#define TEST_HELPERS_CUH_INCLUDED 1

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "helper_cuda.h"

#include "vector_math.cuh"

#include "random_numbers.cuh"

__host__ void uniform_prng_launcher(int num_elements,
                                    curandState *state,
                                    double *h_r);

__global__ void g_uniform_prng(int num_elements,
                               curandState *state,
                               double *r);

__host__ void gaussian_prng(int num_elements,
                           curandState *state,
                           double *h_r);

__global__ void g_gaussian_prng(int num_elements,
                                curandState *state,
                                double *r);

__host__ void gaussian_point(int num_elements,
                             double mean,
                             double std,
                             curandState *state,
                             double3 *h_p);

__global__ void g_gaussian_point(int num_elements,
                                 double mean,
                                 double std,
                                 curandState *state,
                                 double3 *p);

__global__ void zero_elements(int num_elements,
                               double *array);

__global__ void negative_elements(int num_elements,
                                  int2 *array);

__host__ void cu_nan_checker(int num_atoms,
                             double3 *array);

__global__ void g_nan_checker(int num_atoms,
                              double3 *array);

#endif // TEST_HELPERS_CUH_INCLUDED

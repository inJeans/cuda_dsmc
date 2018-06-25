/** \file
 *  \brief Main code
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef RANDOM_NUMBERS_CUH_INCLUDED
#define RANDOM_NUMBERS_CUH_INCLUDED 1

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "dsmc_utils.cuh"
#include "vector_math.cuh"

__device__ double3 dGaussianVector(double mean,
                                   double std,
                                   curandState *state);

__host__ void initRNG(int num_states,
                      int seed,
                      cudaStream_t *streams,
                      curandState **states);

__host__ void cuInitRNG(int num_states,
                        int seed,
                        cudaStream_t stream,
                        curandState *states);

__global__ void gInitRNG(int num_states,
                         int seed,
                         curandState *state);

#endif  // RANDOM_NUMBERS_CUH_INCLUDED

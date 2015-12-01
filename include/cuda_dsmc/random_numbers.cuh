/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <g3log/g3log.hpp>
#include <iostream>
#include <iomanip>

// #include "pcg_variants.h"

// __host__ double h_gaussian_ziggurat(pcg32_ranm,dom_t *seed);

__host__ void cu_initialise_rng_states(int n_states,
                                       curandState *state);

__global__ void g_initialise_rng_states(int n_states,
                                        curandState *state);

__device__ double3 gaussian_point(double mean,
                                  double std,
                                  curandState *seed);

// __host__ double uniform_prng(pcg32_random_t *seed);

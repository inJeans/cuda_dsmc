#include <iostream>
#include <math.h>

#include <cuda_runtime.h>

#include "pcg_variants.h"

__host__ double gaussian_ziggurat(pcg32_random_t *seed);

__host__ __device__ double3 gaussian_point(double mean,
                                           double std,
                                           pcg32_random_t *seed);

__host__ __device__ double uniform_prng(pcg32_random_t *seed);

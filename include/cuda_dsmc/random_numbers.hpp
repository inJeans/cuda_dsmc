/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "random_numbers.cuh"

#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <iostream>

void initialise_rng_states(int n_states,
                           curandState *state);

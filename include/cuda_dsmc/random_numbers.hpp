/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef RANDOM_NUMBERS_HPP_INCLUDED
#define RANDOM_NUMBERS_HPP_INCLUDED 1

#include "random_numbers.cuh"
#include "pcg_variants.h"
#include "entropy.h"

#include <math.h>
#ifdef CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#endif

#include <iostream>

// __host__ double h_gaussian_ziggurat(pcg32_ranm,dom_t *seed);
// __host__ double uniform_prng(pcg32_random_t *seed);

#ifdef CUDA
void initialise_rng_states(int n_states,
                           curandState *state);
#endif

void initialise_rng_states(int n_states,
                           pcg64_random_t *state,
                           bool non_deterministic_seed = true);

void h_initialise_rng_states(bool non_deterministic_seed,
                             int n_states,
                             pcg64_random_t *state);

#endif  // RANDOM_NUMBERS_HPP_INCLUDED

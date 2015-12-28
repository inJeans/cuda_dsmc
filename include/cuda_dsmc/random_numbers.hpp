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

#ifdef CUDA
void initialise_rng_states(int n_states,
                           curandState *state,
                           bool non_deterministic_seed = false);
#endif

void initialise_rng_states(int n_states,
                           pcg32_random_t *state,
                           bool non_deterministic_seed = false);

double3 gaussian_point(double mean,
                       double std,
                       pcg32_random_t *state);

double gaussian_ziggurat(pcg32_random_t *state);

double uniform_prng(pcg32_random_t *state);

#endif  // RANDOM_NUMBERS_HPP_INCLUDED

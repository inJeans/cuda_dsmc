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
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <iostream>

#ifdef CUDA
void initialise_rng_states(int n_states,
                           curandState *state);
#endif

void initialise_rng_states(int n_states,
                           pcg64_random_t *state,
                           bool non_deterministic_seed = true);

double3 gaussian_point(double mean,
                       double std,
                       pcg64_random_t *state);

double gaussian_ziggurat(pcg64_random_t *state);

double uniform_prng(pcg64_random_t *state);

#endif  // RANDOM_NUMBERS_HPP_INCLUDED

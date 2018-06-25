/** \file
 *  \brief Main code
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef RANDOM_NUMBERS_HPP_INCLUDED
#define RANDOM_NUMBERS_HPP_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "include/pcg_variants.h"
#include "extras/entropy.h"  // Wrapper around /dev/random

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/vector_math.cuh"

typedef struct {
    pcg32_random_t gen[2];
} pcg32x2_random_t;

double3 gaussianVector(double mean,
                       double std,
                       pcg32x2_random_t* rng);

double2 boxMuller(pcg32x2_random_t* rng);

double uniformRandom(pcg32x2_random_t* rng);

void pcg32x2_srandom_r(pcg32x2_random_t* rng,
                       uint64_t seed1,
                       uint64_t seed2,
                       uint64_t seq1,
                       uint64_t seq2);

uint64_t pcg32x2_random_r(pcg32x2_random_t* rng);

#endif  // RANDOM_NUMBERS_HPP_INCLUDED
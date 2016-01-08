/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef RANDOM_NUMBER_GENERATION_TESTS_CUH_INCLUDED
#define RANDOM_NUMBER_GENERATION_TESTS_CUH_INCLUDED 1

__host__ void uniform_prng(int num_elements,
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

double mean(double *array,
            int num_elements);

double std_dev(double *array,
               int num_elements);

double z_score(double value,
               double mean,
               double std);

#endif  // RANDOM_NUMBER_GENERATION_TESTS_CUH_INCLUDED
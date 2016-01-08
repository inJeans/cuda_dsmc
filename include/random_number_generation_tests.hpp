/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef RANDOM_NUMBER_GENERATION_TESTS_HPP_INCLUDED
#define RANDOM_NUMBER_GENERATION_TESTS_HPP_INCLUDED 1

double g_uniform_prng(void);

double mean(double *array,
            int num_elements);

double std_dev(double *array,
               int num_elements);

double z_score(double value,
               double mean,
               double std);

#endif  // RANDOM_NUMBER_GENERATION_TESTS_HPP_INCLUDED
/** \file
 *  \brief Unit tests for the distribution_generation file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <cuda_runtime.h>

#include <float.h>
#include <algorithm>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
extern "C"
{
#include "unif01.h"
#include "bbattery.h" 
}

#include "random_numbers.hpp"
#include "random_number_generation_tests.hpp"

pcg32_random_t g_rng;

SCENARIO("[HOST] Uniform random number generation", "[h-urng]") {
    GIVEN("An appropriate seed") {
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, 42u, 54u);

        WHEN("The random number generator is called") {
            double r = uniform_prng(&rng);

            THEN("The result should be between 0 and 1") {
                REQUIRE(r >= 0.);
                REQUIRE(r <= 1.);
            }
        }

        WHEN("We assign the local seed to the global seed") {
            g_rng = rng;
            unif01_Gen *gen;
            char* rng_name = "g_uniform_prng";
            gen = unif01_CreateExternGen01(rng_name,
                                           g_uniform_prng);

            THEN("We expect to pass small crush") {
                bbattery_SmallCrush(gen);
                bool complete = true;
                REQUIRE(complete);
            }

            unif01_DeleteExternGen01(gen);
        }
    }
}

SCENARIO("[HOST] Normally distributed random number generation", "[h-nrng]") {
    GIVEN("An appropriate seed") {
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, 42u, 54u);

        WHEN("We generate 5,000 numbers using a mean of 0 and a std of 1") {
            double test_values[5000];
            for (int i = 0; i < 5000; ++i)
                test_values[i] = gaussian_ziggurat(&rng);

            THEN("The result should pass the back-of-the-envelope test") {
                double val_mean = mean(test_values,
                                       5000);
                double val_std  = std_dev(test_values,
                                          5000);
                double val_max = *std::max_element(test_values,
                                                   test_values+5000);
                double val_min = *std::min_element(test_values,
                                                   test_values+5000);

                double Z_max = z_score(val_max,
                                       val_mean,
                                       val_std);
                double Z_min = z_score(val_min,
                                       val_mean,
                                       val_std);
                REQUIRE(Z_max <= 4.);
                REQUIRE(Z_min >=-4.);
            }

            // Also need to implement a more rigorous test
        }
    }
}

double g_uniform_prng(void) {
    return uniform_prng(&g_rng);
}

double mean(double *array,
            int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i];

    return mean / num_elements;
}

double std_dev(double *array,
               int num_elements) {
    double mu = mean(array,
                     num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i)
        sum += (array[i]-mu) * (array[i]-mu);

    return sqrt(sum / num_elements);
}

double z_score(double value,
               double mean,
               double std) {
    return (value - mean) / std;
}

/** \file
 *  \brief Unit tests for the distribution_generation file
 *
 *  More detailed description
 */

#include <float.h>

#include <cuda_runtime.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "distribution_generation.cuh"
#include "distribution_generation_tests.cuh"

SCENARIO( "Uniform random number generation", "[urng]" ) {

    GIVEN( "An appropriate seed" ) {
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, 42u, 54u);

        WHEN( "The random number generator is called" ) {
            double r = uniform_prng( &rng );

            THEN( "The result should be between 0 and 1" ) {
                REQUIRE( r >= 0. );
                REQUIRE( r <= 1. );
            }
        }
    }
}

SCENARIO( "Normally distributed random number generation", "[nrng]" ) {

    GIVEN( "An appropriate seed" ) {
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, 42u, 54u);

        WHEN( "We generate 5,000 numbers using a mean of 0 and a std of 1" ) {
            double test_values[5000];
            for (int i = 0; i < 5000; ++i)
                test_values[i] = gaussian_point( 0., 
                                                 1., 
                                                 &rng );

            THEN( "The result should pass the back-of-the-envelope test" ) {
                double val_mean = mean( test_values,
                                        5000 );
                double val_std  = std_dev( test_values,
                                           5000 );
                double val_max = max( test_values,
                                      5000 );
                double val_min = min( test_values,
                                      5000 );
                printf( "max = %f, min = %f\n", val_max, val_min );
                printf( "mean = %f, std = %f\n", val_mean, val_std );

                double Z_max = z_score( val_max,
                                        val_mean,
                                        val_std );
                double Z_min = z_score( val_min,
                                        val_mean,
                                        val_std );
                REQUIRE( Z_max <= 4. );
                REQUIRE( Z_min >=-4. );
            }
        }
    }
}

double max( double *array,
            int num_elements )
{
    double max = -1.*(double)DBL_MAX;
    for (int i = 0; i < num_elements; ++i)
    {
        if ( array[i] > max ) max = array[i];
    }
    return max;
}

double min( double *array,
            int num_elements )
{
    double min = (double)DBL_MAX;
    for (int i = 0; i < num_elements; ++i)
    {
        if ( array[i] < min ) min = array[i];
    }

    return min;
}

double mean( double *array,
             int num_elements )
{
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
    {
        mean += array[i];
    }

    return mean / num_elements;
}

double std_dev( double *array,
            int num_elements )
{
    double mu = mean( array,
                      num_elements );
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i)
    {
        sum += (array[i]-mu) * (array[i]-mu);
    }

    return sqrt( sum / num_elements );
}

double z_score( double value,
                double mean,
                double std )
{
    return (value - mean) / std;
}
/** \file
 *  \brief Unit tests for the distribution_generation file
 *
 *  More detailed description
 */

#include <float.h>

#include <cuda_runtime.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
extern "C"
{
#include "unif01.h"
#include "bbattery.h" 
}

#include "distribution_generation.cuh"

pcg32_random_t g_rng;

double g_uniform_prng( void )
{
    return uniform_prng( &g_rng );
}

SCENARIO( "Uniform random number generation", "[urng]" ) {

    GIVEN( "An appropriate seed" ) {
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, 42u, 54u);

        WHEN( "We assign the local seed to the global seed" ) {
            g_rng = rng;
            unif01_Gen *gen;
            char* rng_name = "g_uniform_prng";
            gen = unif01_CreateExternGen01( rng_name,
                                            g_uniform_prng );

            THEN( "We expect to pass crush" ) {
                bbattery_Crush( gen );
                bool complete = true;
                REQUIRE( complete );
            }
        }
    }
}
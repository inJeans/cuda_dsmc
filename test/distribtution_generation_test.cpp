/** \file
 *  \brief Unit tests for the distribution_generation file
 *
 *  More detailed description
 */

#include <cuda_runtime.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "distribution_generation.cuh"

SCENARIO( "Gaussian distributed point can be generated", "[gauss]" ) {

    GIVEN( "A mean and a standard deviation" ) {
        double mean;
        double std;

        // REQUIRE( v.size() == 5 );

        WHEN( "The mean and standard deviation are both zero" ) {
            mean = 0.;
            std = 0.;

            THEN( "The result should be between 0 and 1" ) {
                double r = gaussian_point( 0., 0., 0. );
                REQUIRE( r >= 0. );
                REQUIRE( r <= 1. );
            }
        }
    }
}
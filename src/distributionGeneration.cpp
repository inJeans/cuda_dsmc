/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 */

#include <iostream>
#include <math.h>

#include <cuda_runtime.h>

#include "distribution_generation.cuh"
#include "pcg_variants.h"

 __host__ __device__ double gaussian_point( double mean,
 	                                        double std,
 	                                        double seed )
{
	double r = uniform_prng( 0 );

 	return r;
}

__host__ __device__ double uniform_prng( int seed )
{
	pcg64_random_t rng;
    pcg64_srandom_r(&rng, 42u, 54u);
    double r = ldexp(pcg64_random_r(&rng), -64);

	return r;
}
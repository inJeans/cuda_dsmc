/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 */

#include <iostream>
#include <random>

#include <cuda_runtime.h>

#include "distribution_generation.cuh"

 __host__ __device__ double gaussian_point( double mean,
 	                                        double std,
 	                                        double seed )
{
	double r = uniform_prng( 0 );

 	return r;
}

__host__ __device__ double uniform_prng( int seed )
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    double r = dis( gen );

	return r;
}
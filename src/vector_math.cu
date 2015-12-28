/** \file
 *  \brief Definition of the trapping potential
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "vector_math.cuh"

__host__ __device__ double norm(double3 vec) {
	return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
/** \file
 *  \brief Definition of the trapping potential
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "vector_math.cuh"

__host__ __device__ double dot(double3 a, double3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ double3 unit(double3 vec) {
    return vec / norm(vec);
}

__host__ __device__ double norm(double3 vec) {
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

__host__ __device__ double3 floor(double3 vec) {
    return make_double3(floor(vec.x),
                        floor(vec.y),
                        floor(vec.z));
}

__host__ __device__ int3 type_cast_int3(double3 vec) {
    return make_int3(static_cast<int>(vec.x),
                     static_cast<int>(vec.y),
                     static_cast<int>(vec.z));
}

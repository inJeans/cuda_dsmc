/** \file
 *  \brief Definition of the trapping potential
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "trapping_potential.cuh"

 __host__ __device__ double3 B(double3 r,
                               trap_geo params) {
    double3 mag_field = make_double3( 0., 0., 0.);

    mag_field.x = 0.5 * params.Bz * r.x;
    mag_field.y = 0.5 * params.Bz * r.y;
    mag_field.z =-1.0 * params.Bz * r.z;

    return mag_field;
 }

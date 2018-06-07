/** \file
 *  \brief Random number generators and associated random functions
 *
 *  Here we define all the functions required for random number generation,
 *  from the generation of seeds and initialisation of rngs to the generation
 *  of random vectors in cartesian space.
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "cuda_dsmc/magnetic_field.hpp"

#if defined (HARMONIC)
const double kMaxDistributionWidth = 1.;

/** \brief Generates the magnetic field for a given position
 *
 *  \param params Structure containing the parameters that describe the 
 *  magnetic field.
 *  \param pos A double3 describing the position.
 *  \exception not yet.
 *  \return A double3 containing the magnetic field vector at the
 *  described position
 */
double3 magneticField(FieldParams params,
                      double3 pos) {
    double3 omega = params.omega;
    double3 magnetic_field = make_double3(0., 0., 0.);

    magnetic_field.x = omega.x * omega.x * pos.x * pos.x;
    magnetic_field.y = omega.y * omega.y * pos.y * pos.y;
    magnetic_field.z = omega.z * omega.z * pos.z * pos.z;

    return magnetic_field;
}

#else  // No magnetic field
const double kMaxDistributionWidth = 1.;

/** \brief Generates the magnetic field for a given position
 *
 *  \param params Structure containing the parameters that describe the 
 *  magnetic field.
 *  \param pos A double3 describing the position.
 *  \exception not yet.
 *  \return A double3 containing the magnetic field vector at the
 *  described position
 */
double3 magneticField(FieldParams params,
                      double3 pos) {
    double3 magnetic_field = make_double3(0., 0., 0.);

    if (pos.x > kMaxDistributionWidth || pos.x < -kMaxDistributionWidth) {
        magnetic_field.x = 1.e99;
    } else {
        magnetic_field.x = 0.;
    }
    if (pos.y > kMaxDistributionWidth || pos.y < -kMaxDistributionWidth) {
        magnetic_field.y = 1.e99;
    } else {
        magnetic_field.y = 0.;
    }
    if (pos.z > kMaxDistributionWidth || pos.z < -kMaxDistributionWidth) {
        magnetic_field.z = 1.e99;
    } else {
        magnetic_field.z = 0.;
    }

    return magnetic_field;
}

#endif

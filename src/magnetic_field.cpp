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

    magnetic_field.x = 0.5 * omega.x * omega.x * pos.x * pos.x;
    magnetic_field.y = 0.5 * omega.y * omega.y * pos.y * pos.y;
    magnetic_field.z = 0.5 * omega.z * omega.z * pos.z * pos.z;

    return magnetic_field;
}

/** \brief Generates the gradient magnetic field for a given position
 *
 *  \param params Structure containing the parameters that describe the 
 *  magnetic field.
 *  \param pos A double3 describing the position.
 *  \exception not yet.
 *  \return A double3 containing the magnetic field gradient vector at 
 *  the described position
 */
double3 magneticFieldGradient(FieldParams params,
                              double3 pos) {
    double3 omega = params.omega;
    double3 magnetic_field_gradient = make_double3(0., 0., 0.);

    magnetic_field_gradient.x = omega.x * omega.x * pos.x;
    magnetic_field_gradient.y = omega.y * omega.y * pos.y;
    magnetic_field_gradient.z = omega.z * omega.z * pos.z;

    return magnetic_field_gradient;
}

#else  // No magnetic field

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
    // TODO(Chris): Change to linear with steep gradient
    double3 magnetic_field = make_double3(0., 0., 0.);
    double max_distribution_width = params.max_distribution_width;

    if (pos.x > max_distribution_width || pos.x < -max_distribution_width) {
        magnetic_field.x = 1.e99;
    } else {
        magnetic_field.x = 0.;
    }
    if (pos.y > max_distribution_width || pos.y < -max_distribution_width) {
        magnetic_field.y = 1.e99;
    } else {
        magnetic_field.y = 0.;
    }
    if (pos.z > max_distribution_width || pos.z < -max_distribution_width) {
        magnetic_field.z = 1.e99;
    } else {
        magnetic_field.z = 0.;
    }

    return magnetic_field;
}

/** \brief Generates the magnetic field gradient for a given position
 *
 *  \param params Structure containing the parameters that describe the 
 *  magnetic field.
 *  \param pos A double3 describing the position.
 *  \exception not yet.
 *  \return A double3 containing the magnetic field vector at the
 *  described position
 */
double3 magneticFieldGradient(FieldParams params,
                              double3 pos) {
    double3 magnetic_field_gradient = make_double3(0., 0., 0.);
    double max_distribution_width = params.max_distribution_width;

    if (pos.x > max_distribution_width || pos.x < -max_distribution_width) {
        magnetic_field_gradient.x = 1.e99;
    } else {
        magnetic_field_gradient.x = 0.;
    }
    if (pos.y > max_distribution_width || pos.y < -max_distribution_width) {
        magnetic_field_gradient.y = 1.e99;
    } else {
        magnetic_field_gradient.y = 0.;
    }
    if (pos.z > max_distribution_width || pos.z < -max_distribution_width) {
        magnetic_field_gradient.z = 1.e99;
    } else {
        magnetic_field_gradient.z = 0.;
    }

    return magnetic_field_gradient;
}

#endif

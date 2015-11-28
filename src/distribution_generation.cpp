/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_generation.hpp"
#include "distribution_generation.cuh"

/** \fn void generate_thermal_velocities(int num_atoms,
 *                                       double temp,
 *                                       curandState *state,
                                         double3 *vel) 
 *  \brief Calls the function to fill an array of thermal velocties at
 *  temperature temp.
 *  \param mean Gaussian mean
 *  \exception not yet.
 *  \return a gaussian distributed point in cartesian space
*/

void generate_thermal_velocities(int num_atoms,
                                 double temp,
                                 curandState *state,
                                 double3 *vel) {
#ifdef CUDA
    cu_generate_thermal_velocities(num_atoms,
                                   temp,
                                   state,
                                   vel);
#endif

    return;
}

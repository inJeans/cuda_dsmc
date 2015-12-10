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
    cu_generate_thermal_velocities(num_atoms,
                                   temp,
                                   state,
                                   vel);

    return;
}

// /** \fn void generate_thermal_velocities(int num_atoms,
//  *                                       double temp,
//  *                                       pcg64_random_t *state,
//                                          double3 *vel) 
//  *  \brief Calls the function to fill an array of thermal velocties at
//  *  temperature temp.
//  *  \param mean Gaussian mean
//  *  \exception not yet.
//  *  \return a gaussian distributed point in cartesian space
// */

// void generate_thermal_velocities(int num_atoms,
//                                  double temp,
//                                  pcg64_random_t *state,
//                                  double3 *vel) {
//     h_generate_thermal_velocities(num_atoms,
//                                   temp,
//                                   state,
//                                   vel);

//     return;
// }

// * \fn __global__ void h_generate_thermal_velocities(int num_atoms,
//  *                                                    double temp,
//  *                                                    pcg64_random_t *state,
//  *                                                    double3 *vel) 
//  *  \brief description
//  *  \param num_atoms Total number of atoms in the gas.
//  *  \param temp Temperature of the gas (in Kelvin).
//  *  \param *seed Pointer to an array of seeds for the random number generator
//  *  \param *vel Pointer to an output array of length num_atoms for storing
//     the gas velocities.
//  *  \exception not yet.
//  *  \return void


// __global__ void h_generate_thermal_velocities(int num_atoms,
//                                               double temp,
//                                               pcg64_random_t *state,
//                                               double3 *vel) {
//     for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
//          atom < num_atoms;
//          atom += blockDim.x * gridDim.x) {
//         vel[atom] = d_thermal_vel(temp,
//                                   &state[atom]);
//     }

//     return;
// }

// /** \fn __host__ __device__ double3 d_thermal_vel(double temp,
//  *                                                pcg64_random_t *state) 
//  *  \brief description
//  *  \param temp Temperature of the gas (in Kelvin).
//  *  \param *seed Pointer to a seed for the random number generator.
//  *  \exception not yet.
//  *  \return a gaussian distributed point in cartesian space with the standard
//     deviation expected for a thermal gas
// */

// __device__ double3 d_thermal_vel(double temp,
//                                  pcg64_random_t *state) {
//     double V = sqrt(d_kB * temp / d_mass);
//     double3 vel = gaussian_point(0,
//                                  V,
//                                  state);
//     return vel;
// }

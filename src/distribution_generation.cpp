/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_generation.hpp"
#ifdef CUDA
#include "distribution_generation.cuh"
#endif

const double gs   =  0.5;             // Gyromagnetic ratio
const double MF   = -1.0;             // Magnetic quantum number
const double muB  = 9.27400915e-24;   // Bohr magneton
const double mass = 1.443160648e-25;  // 87Rb mass
const double pi   = 3.14159265;       // Pi
const double a    = 5.3e-9;           // Constant cross-section formula
const double kB   = 1.3806503e-23;    // Boltzmann's Constant
const double hbar = 1.05457148e-34;   // hbar

/** \fn void generate_thermal_velocities(int num_atoms,
 *                                       double temp,
 *                                       curandState *state,
                                         double3 *vel) 
 *  \brief Calls the function to fill a `double3` array of thermal velocties 
 *  on the device with a mean temperature of `temp`.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param *state Pointer to a `curandState` device array of length `num_atoms`.
 *  \param *vel Pointer to a `double3` device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

#ifdef CUDA
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
#endif

/** \fn void generate_thermal_velocities(int num_atoms,
 *                                       double temp,
 *                                       pcg32_random_t *state,
 *                                       double3 *vel)
 *  \brief Calls the function to fill a `double3` array of thermal velocties 
 *  on the host with a mean temperature of `temp`.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param *state Pointer to a `pcg32_random_t` host array of length `num_atoms`.
 *  \param *vel Pointer to a `double3` host array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

void generate_thermal_velocities(int num_atoms,
                                 double temp,
                                 pcg32_random_t *state,
                                 double3 *vel) {
    for (int atom = 0; atom < num_atoms; ++atom) {
        vel[atom] = thermal_vel(temp,
                                &state[atom]);
    }

    return;
}

/** \fn thermal_vel(double temp,
                      pcg32_random_t *state)
 *  \brief Calls the function to generate a `double3` thermal velocty on the
 *  host with a mean temperature of `temp`.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param *state Pointer to a single `pcg32_random_t` state on the host.
 *  \exception not yet.
 *  \return void
*/

double3 thermal_vel(double temp,
                    pcg32_random_t *state) {
    double V = sqrt(kB * temp / mass);
    double3 vel = gaussian_point(0,
                                 V,
                                 state);
    return vel;
}

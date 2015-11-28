/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_generation.cuh"

__constant__ double d_gs   =  0.5;      // Gyromagnetic ratio
__constant__ double d_MF   = -1.0;      // Magnetic quantum number
__constant__ double d_muB  = 9.27400915e-24;  // Bohr magneton
__constant__ double d_mass = 1.443160648e-25;// 87Rb mass
__constant__ double d_pi   = 3.14159265;    // Pi
__constant__ double d_a    = 5.3e-9;      // Constant cross-section formula
__constant__ double d_kB   = 1.3806503e-23; // Boltzmann's Constant
__constant__ double d_hbar = 1.05457148e-34;  // hbar

/** \fn __host__ void cu_generate_thermal_velocities(int num_atoms,
 *                                                   double temp,
 *                                                   curandState *state,
                                                     double3 *vel) 
 *  \brief Calls the function to fill an array of thermal velocties at
 *  temperature temp.
 *  \param mean Gaussian mean
 *  \exception not yet.
 *  \return a gaussian distributed point in cartesian space
*/

__host__ void cu_generate_thermal_velocities(int num_atoms,
                                             double temp,
                                             curandState *state,
                                             double3 *vel) {
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_generate_thermal_velocities,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;

    g_generate_thermal_velocities<<<grid_size,
                                    block_size>>>
                                 (num_atoms,
                                  temp,
                                  state,
                                  vel);  

    return;
}

/** \fn __global__ void g_generate_thermal_velocities(int num_atoms,
 *                                                    double temp,
 *                                                    curandState *state,
 *                                                    double3 *vel) 
 *  \brief description
 *  \param num_atoms Total number of atoms in the gas.
 *  \param temp Temperature of the gas (in Kelvin).
 *  \param *seed Pointer to an array of seeds for the random number generator
 *  \param *vel Pointer to an output array of length num_atoms for storing
    the gas velocities.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_generate_thermal_velocities(int num_atoms,
                                              double temp,
                                              curandState *state,
                                              double3 *vel) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        vel[atom] = d_thermal_vel(temp,
                                  &state[atom]);
    }

    return;
}

/** \fn __host__ __device__ double3 d_thermal_vel(double temp,
 *                                                curandState *state) 
 *  \brief description
 *  \param temp Temperature of the gas (in Kelvin).
 *  \param *seed Pointer to a seed for the random number generator.
 *  \exception not yet.
 *  \return a gaussian distributed point in cartesian space with the standard
    deviation expected for a thermal gas
*/

__device__ double3 d_thermal_vel(double temp,
                                 curandState *state) {
    double V = sqrt(d_kB * temp / d_mass);
    double3 vel = gaussian_point(0,
                                 V,
                                 state);
    return vel;
}

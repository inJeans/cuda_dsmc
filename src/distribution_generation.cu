/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_generation.cuh"

#include "declare_device_constants.cuh"

__constant__ double d_max_grid_width = 2.e-3;

/** \fn __host__ void cu_generate_aligned_spins(int num_atoms,
 *                                              trap_geo params,
 *                                              double3 *pos,
 *                                              zomplex2 *psi)
 *  \brief Calls the `__global__` function to fill a `zomplex2` array of aligned spins 
 *  on the device with a mean projection of 1.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param params (TODO).
 *  \param *pos Pointer to a `double3` device array of length `num_atoms`.
 *  \param *zomplex2 Pointer to a `zomplex2` device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_generate_aligned_spins(int num_atoms,
                                        trap_geo params,
                                        double3 *pos,
                                        zomplex2 *psi) {
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the wavefunction "
                "initialisation kernel.\n");
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_generate_aligned_spins,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
    g_generate_aligned_spins<<<grid_size,
                               block_size>>>
                            (num_atoms,
                             params,
                             pos,
                             psi);

    return;
}

/** \fn __global__ void g_generate_aligned_spins(int num_atoms,
 *                                               trap_geo params,
 *                                               double3 *pos,
 *                                               zomplex2 *psi)
 *  \brief `__global__` function for filling a `double3` array of length
 *  `num_atoms` with a distribution of aligned spins.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param params (TODO).
 *  \param *pos Pointer to a `double3` device array of length `num_atoms`.
 *  \param *zomplex2 Pointer to a `zomplex2` device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_generate_aligned_spins(int num_atoms,
                                         trap_geo params,
                                         double3 *pos,
                                         zomplex2 *psi) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        psi[atom] = d_aligned_spin(params,
                                   pos[atom]);
    }

    return;
}

/** \fn __host__ __device__ zomplex2 aligned_spin(trap_geo params,
  *                                               double3 pos) 
 *  \brief `__device__` function for generating a single aligned spin
 *  given a magnetic field and a position.
 *  \param params TODO.
 *  \param pos A `double3` element describing the position of the particle.
 *  \exception not yet.
 *  \return A `cuDoubleComplex2` element representing the wavefunction of the 
 *  particle.
*/

__device__ zomplex2 d_aligned_spin(trap_geo params,
                                   double3 pos) {
    zomplex2 psi = make_zomplex2(0., 0., 0., 0.);

    double3 Bn = unit(B(pos,
                        params));

    psi.up.x = 0.5 * (1. + Bn.x + Bn.z) / sqrt(1 + Bn.x);
    psi.up.y = 0.5 * (-1.*Bn.y) / sqrt(1 + Bn.x);
    psi.dn.x = 0.5 * (1. + Bn.x - Bn.z) / sqrt(1 + Bn.x);
    psi.dn.y = 0.5 * (Bn.y) / sqrt(1 + Bn.x);

    return psi;
}

/** \fn __host__ void cu_generate_thermal_velocities(int num_atoms,
 *                                                   double temp,
 *                                                   curandState *state,
                                                     double3 *vel) 
 *  \brief Calls the `__global__` function to fill an array of thermal 
 *  velocties with a mean temperature of `temp`.
 *  \param temp Mean temperature of the thermal gas, as defined by (TODO).
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_generate_thermal_velocities(int num_atoms,
                                             double temp,
                                             curandState *state,
                                             double3 *vel) {
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the velocity "
                "initialisation kernel.\n");
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_generate_thermal_velocities,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);

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
 *  \brief `__global__` function for filling a `double3` array of length
 *  `num_atoms` with a distribution of thermal velocities.
 *  \param num_atoms Total number of atoms in the gas.
 *  \param temp Temperature of the gas (in Kelvin).
 *  \param *state Pointer to an array of `curandState` states for the random
 *  number generator
 *  \param *vel Pointer to an output `double3` array of length `num_atoms` for
 *  storing the gas velocities.
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

/** \fn __device__ double3 d_thermal_vel(double temp,
 *                                       curandState *state) 
 *  \brief `__device__` function for generating a single thermal velocity
 *  given a temperature `temp`.
 *  \param temp Mean temperature of the gas (in Kelvin).
 *  \param *state Pointer to a `curandState` state for the random number
 *  generator.
 *  \exception not yet.
 *  \return a gaussian distributed point in cartesian space with the standard
 *  deviation expected for a thermal gas as given in (TODO).
*/

__device__ double3 d_thermal_vel(double temp,
                                 curandState *state) {
    double V = sqrt(d_kB * temp / d_mass);
    double3 vel = d_gaussian_point(0,
                                   V,
                                   state);
    return vel;
}

/** \fn __host__ void cu_generate_thermal_positions(int num_atoms,
 *                                                  double temp,
 *                                                  curandState *state,
 *                                                  double3 *pos) 
 *  \brief Calls the `__global__` function to fill an array of thermal 
 *  velocties with a mean temperature of `temp`.
 *  \param temp Mean temperature of the thermal gas, as defined by (TODO).
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_generate_thermal_positions(int num_atoms,
                                            double temp,
                                            trap_geo params,
                                            curandState *state,
                                            double3 *pos) {
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the velocity "
                "initialisation kernel.\n");
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_generate_thermal_positions,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);

    g_generate_thermal_positions<<<grid_size,
                                    block_size>>>
                                 (num_atoms,
                                  temp,
                                  params,
                                  state,
                                  pos);  

    return;
}

/** \fn __global__ void g_generate_thermal_positions(int num_atoms,
 *                                                   double temp,
 *                                                   trap_geo params,
 *                                                   curandState *state,
 *                                                   double3 *pos)
 *  \brief Calls the function to fill a `double3` array of thermal positions 
 *  on the host with a distribution determined by the trapping potential.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param params TODO
 *  \param *state Pointer to a `curandState` host array of length `num_atoms`.
 *  \param *pos Pointer to a `double3` host array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_generate_thermal_positions(int num_atoms,
                                             double temp,
                                             trap_geo params,
                                             curandState *state,
                                             double3 *pos) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        pos[atom] = thermal_pos(temp,
                                params,
                                &state[atom]);
    }

    return;
}

/** \fn __device__ thermal_pos(double temp,
 *                             trap_geo params,
 *                             scurandState *state)
 *  \brief Calls the function to generate a `double3` thermal pos on the
 *  host with a distribution determined by the trapping potential.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param params TODO
 *  \param *state Pointer to a single `curandState` state on the host.
 *  \exception not yet.
 *  \return void
*/

__device__ double3 thermal_pos(double temp,
                               trap_geo params,
                               curandState *state) {
    bool no_atom_selected = true;
    double3 pos = make_double3(0., 0., 0.);

    while (no_atom_selected) {
        double3 r = d_gaussian_point(0.,
                                     1.,
                                     state);
        r = r * d_max_grid_width;

        double magB = norm(B(r,
                             params));
        double U = 0.5 * (magB - params.B0) * d_gs * d_muB;
        double Pr = exp(-U / d_kB / temp);

        if (curand_uniform(state) < Pr) {
            pos = r;
            no_atom_selected = false;
        }
    }

    return pos;
}

/** \fn __host__ void cu_initialise_atom_id(int num_atoms,
 *                                          int *atom_id) 
 *  \brief Calls the `__global__` function to fill an array of thermal 
 *  velocties with a mean temperature of `temp`.
 *  \param temp Mean temperature of the thermal gas, as defined by (TODO).
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_initialise_atom_id(int num_atoms,
                                    int *atom_id) {
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the atom_id "
                "initialisation kernel.\n");
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_initialise_atom_id,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);

    g_initialise_atom_id<<<grid_size,
                           block_size>>>
                          (num_atoms,
                           atom_id);  

    return;
}

/** \fn __global__ void g_initialise_atom_id(int num_atoms,
 *                                           int *atom_id)
 *  \brief Calls the function to fill a `double3` array of thermal positions 
 *  on the host with a distribution determined by the trapping potential.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param params TODO
 *  \param *state Pointer to a `curandState` host array of length `num_atoms`.
 *  \param *pos Pointer to a `double3` host array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_initialise_atom_id(int num_atoms,
                                     int *atom_id) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        atom_id[atom] = atom;
    }

    return;
}

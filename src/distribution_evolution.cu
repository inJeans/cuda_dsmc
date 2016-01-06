/** \file
 *  \brief Device functions for distribution evolution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_evolution.cuh"

#include "declare_device_constants.cuh"

/** \fn __host__ void cu_update_atom_accelerations(int num_atoms,
 *                                                 trap_geo params,
 *                                                 double3 *pos,
 *                                                 double3 *acc)
 *  \brief Calls the `__global__` function to fill an array with accelerations 
 *  based on their position and the trapping potential.
 *  \param num_atoms Total number of atoms in the gas.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos A `double3` array of length `num_atoms` containing the position
 *  of each atom.
 *  \param *acc A `double3` array of length `num_atoms` containing the
 *  acceleration of each atom.
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_update_atom_accelerations(int num_atoms,
                                           trap_geo params,
                                           double3 *pos,
                                           double3 *acc) {
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the acceleration "
                "update kernel.\n");
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_update_atom_accelerations,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);

    g_update_atom_accelerations<<<grid_size,
                                  block_size>>>
                                 (num_atoms,
                                  params,
                                  pos,
                                  acc);  

    return;
}

/** \fn __global__ void g_generate_thermal_velocities(int num_atoms,
 *                                                    trap_geo params,
 *                                                    double3 *pos,
 *                                                    double3 *acc)
 *  \brief `__global__` function for filling a `double3` array of length
 *  `num_atoms` with accelerations based on their position and the trapping 
 *  potential.
 *  \param num_atoms Total number of atoms in the gas.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos Pointer to an input `double3` array of length `num_atoms` for
 *  storing the gas positions.
 *  \param *acc Pointer to an output `double3` array of length `num_atoms` for
 *  storing the gas accelerations.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_update_atom_accelerations(int num_atoms,
                                            trap_geo params,
                                            double3 *pos,
                                            double3 *acc) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        acc[atom] = d_update_acceleration(pos[atom],
                                          params);
    }

    return;
}

__device__ double3 d_update_acceleration(double3 pos,
                                         trap_geo params) {
    double3 acc = make_double3(0., 0., 0.);

    acc.x = d_dV_dx(pos,
                    params) / d_mass;
    acc.y = d_dV_dy(pos,
                    params) / d_mass;
    acc.z = d_dV_dz(pos,
                    params) / d_mass;

    return acc;
}
/** \file
 *  \brief Device functions for distribution evolution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "cublas_v2.h"

#include "distribution_evolution.cuh"

#include "declare_device_constants.cuh"

__host__ void cu_update_positions(int num_atoms,
                                  double dt,
                                  cublasHandle_t cublas_handle,
                                  double3 *vel,
                                  double3 *pos) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunching cuBLAS Daxpy.\n"); 
#endif

    checkCudaErrors(cublasDaxpy(cublas_handle,
                                3*num_atoms,
                                &dt,
                                reinterpret_cast<double *>(vel),
                                1,
                                reinterpret_cast<double *>(pos),
                                1));

    return;
}

__global__ void g_update_atom_position(int num_atoms,
                                       double dt,
                                       double3 *vel,
                                       double3 *pos) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        pos[atom] = d_update_atom_position(dt,
                                           pos[atom],
                                           vel[atom]);
    }

    return;
}

__device__ double3 d_update_atom_position(double dt,
                                          double3 pos,
                                          double3 vel) {
    return pos + vel * dt;
}

__host__ void cu_update_velocities(int num_atoms,
                                   double dt,
                                   cublasHandle_t cublas_handle,
                                   double3 *acc,
                                   double3 *vel) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunching cuBLAS Daxpy.\n");
#endif

    checkCudaErrors(cublasDaxpy(cublas_handle,
                                3*num_atoms,
                                &dt,
                                reinterpret_cast<double *>(acc),
                                1,
                                reinterpret_cast<double *>(vel),
                                1));

    return;
}

__global__ void g_update_atom_velocity(int num_atoms,
                                       double dt,
                                       double3 *acc,
                                       double3 *vel) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        vel[atom] = d_update_atom_velocity(dt,
                                           vel[atom],
                                           acc[atom]);
    }

    return;
}

__device__ double3 d_update_atom_velocity(double dt,
                                          double3 vel,
                                          double3 acc) {
    return vel + acc * dt;
}

/** \fn __host__ void cu_update_accelerations(int num_atoms,
 *                                            trap_geo params,
 *                                            double3 *pos,
 *                                            wavefunction *psi,
 *                                            double3 *acc)
 *  \brief Calls the `__global__` function to fill an array with accelerations 
 *  based on their position and the trapping potential.
 *  \param num_atoms Total number of atoms in the gas.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos A `double3` array of length `num_atoms` containing the position
 *  of each atom.
 *  \param *psi A `wavefunction` array of length `num_atoms` containing the 
 *  wavefunction of each atom.
 *  \param *acc A `double3` array of length `num_atoms` containing the
 *  acceleration of each atom.
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_update_accelerations(int num_atoms,
                                      trap_geo params,
                                      double3 *pos,
                                      wavefunction *psi,
                                      double3 *acc) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the acceleration "
                "update kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_update_atom_acceleration,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif

    g_update_atom_acceleration<<<grid_size,
                                 block_size>>>
                                (num_atoms,
                                 params,
                                 pos,
                                 psi,
                                 acc);

    return;
}

/** \fn __global__ void g_update_atom_acceleration(int num_atoms,
 *                                                 trap_geo params,
 *                                                 double3 *pos,
 *                                                 wavefunction *psi,
 *                                                 double3 *acc)
 *  \brief `__global__` function for filling a `double3` array of length
 *  `num_atoms` with accelerations based on their position and the trapping 
 *  potential.
 *  \param num_atoms Total number of atoms in the gas.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos Pointer to an input `double3` array of length `num_atoms` for
 *  storing the gas positions.
 *  \param *psi Pointer to an input `wavefunction` array of length `num_atoms` for
 *  storing the gas wavefunctions.
 *  \param *acc Pointer to an output `double3` array of length `num_atoms` for
 *  storing the gas accelerations.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_update_atom_acceleration(int num_atoms,
                                           trap_geo params,
                                           double3 *pos,
                                           wavefunction *psi,
                                           double3 *acc) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
#if defined(SPIN)
        acc[atom] = d_update_atom_acceleration(params,
                                               pos[atom],
                                               psi[atom]);
#else
        acc[atom] = d_update_atom_acceleration(params,
                                               pos[atom]);
#endif
    }

    return;
}

__device__ double3 d_update_atom_acceleration(trap_geo params,
                                              double3 pos) {
    double3 acc = make_double3(0., 0., 0.);

    acc.x = d_dV_dx(pos,
                    params) / d_mass;
    acc.y = d_dV_dy(pos,
                    params) / d_mass;
    acc.z = d_dV_dz(pos,
                    params) / d_mass;

    return acc;
}

__device__ double3 d_update_atom_acceleration(trap_geo params,
                                              double3 pos,
                                              wavefunction psi) {
    double3 acc = make_double3(0., 0., 0.);
    zomplex2 l_psi = make_zomplex2(psi.up.x, psi.up.y,
                                   psi.dn.x, psi.dn.y);

    acc.x = d_expectation_dV_dx(params,
                                pos,
                                l_psi) / d_mass;
    acc.y = d_expectation_dV_dy(params,
                                pos,
                                l_psi) / d_mass;
    acc.z = d_expectation_dV_dz(params,
                                pos,
                                l_psi) / d_mass;
    return acc;
}

/** \fn __host__ void cu_update_wavefunctions(int num_atoms,
 *                                            double dt, 
 *                                            trap_geo params,
 *                                            double3 *pos,
 *                                            wavefunction *psi)
 *  \brief Calls the `__global__` function to TODO.
 *  \param num_atoms Total number of atoms in the gas.
 *  \param dt Timestep over which to evolve system.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos A `double3` array of length `num_atoms` containing the position
 *  of each atom.
 *  \param *psi A `wavefunction` array of length `num_atoms` containing the
 *  acceleration of each atom.
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_update_wavefunctions(int num_atoms,
                                      double dt,
                                      trap_geo params,
                                      double3 *pos,
                                      wavefunction *psi) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the wavefunction "
                "update kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_update_atom_wavefunction,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif

    g_update_atom_wavefunction<<<grid_size,
                                 block_size>>>
                                (num_atoms,
                                 dt,
                                 params,
                                 pos,
                                 psi);  

    return;
}

/** \fn __global__ void g_update_atom_wavefunction(int num_atoms,
 *                                                 double dt,
 *                                                 trap_geo params,
 *                                                 double3 *pos,
 *                                                 wavefunction *psi)
 *  \brief `__global__` function for filling a `double3` array of length
 *  `num_atoms` TODO.
 *  \param num_atoms Total number of atoms in the gas.
 *  \params dt Timestep over which to evolve the system.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos Pointer to an input `double3` array of length `num_atoms` for
 *  storing the gas positions.
 *  \param *psi Pointer to an output `wavefunction` array of length `num_atoms` for
 *  storing the gas wavefunctions.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_update_atom_wavefunction(int num_atoms,
                                           double dt,
                                           trap_geo params,
                                           double3 *pos,
                                           wavefunction *psi) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        psi[atom] = d_update_atom_wavefunction(dt,
                                               params,
                                               pos[atom],
                                               psi[atom]);
    }

    return;
}

__device__ wavefunction d_update_atom_wavefunction(double dt,
                                                   trap_geo params,
                                                   double3 pos,
                                                   wavefunction psi) {
    double3 mag_field = B(pos,
                          params);
    double3 Bn = unit(mag_field);
    double norm_B = norm(mag_field);

    double delta_theta = 0.5*d_gs*d_muB*norm_B*dt / d_hbar;
    double cos_delta_theta = cos(delta_theta);
    double sin_delta_theta = sin(delta_theta);

    cuDoubleComplex U[2][2] = {make_cuDoubleComplex(0., 0.)};
    U[0][0] = make_cuDoubleComplex(cos_delta_theta,
                                   -Bn.z*sin_delta_theta);
    U[0][1] = make_cuDoubleComplex(Bn.y*sin_delta_theta,
                                   -Bn.x*sin_delta_theta);
    U[1][0] = make_cuDoubleComplex(-Bn.y*sin_delta_theta,
                                   -Bn.x*sin_delta_theta);
    U[1][1] = make_cuDoubleComplex(cos_delta_theta,
                                   Bn.z*sin_delta_theta);

    wavefunction updated_psi = make_wavefunction(0., 0., 0., 0., psi.isSpinUp);
    updated_psi.up = U[0][0]*psi.up + U[0][1]*psi.dn;
    updated_psi.dn = U[1][0]*psi.up + U[1][1]*psi.dn;

    return updated_psi;
}

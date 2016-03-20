/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DISTRIBUTION_GENERATION_CUH_INCLUDED
#define DISTRIBUTION_GENERATION_CUH_INCLUDED 1

#include <cuda_runtime.h>
#if defined(LOGGING)
#include <g3log/g3log.hpp>
#endif

#include "trapping_potential.cuh"
#include "random_numbers.hpp"
#include "vector_math.cuh"

__host__ void cu_generate_aligned_spins(int num_atoms,
                                        trap_geo params,
                                        double3 *pos,
                                        wavefunction *psi);

__global__ void g_generate_aligned_spins(int num_atoms,
                                         trap_geo params,
                                         double3 *pos,
                                         wavefunction *psi);

__device__ wavefunction d_aligned_wavefunction(trap_geo params,
                                               double3 pos);

__device__ zomplex2 d_aligned_spin(trap_geo params,
                                   double3 pos);

__host__ void cu_generate_thermal_velocities(int num_atoms,
                                             double temp,
                                             curandState *state,
                                             double3 *vel);

__global__ void g_generate_thermal_velocities(int num_atoms,
                                              double temp,
                                              curandState *state,
                                              double3 *vel);

__device__ double3 d_thermal_vel(double temp,
                                 curandState *state);

__host__ void cu_generate_thermal_positions(int num_atoms,
                                            double temp,
                                            trap_geo params,
                                            curandState *state,
                                            double3 *pos);

__global__ void g_generate_thermal_positions(int num_atoms,
                                             double temp,
                                             trap_geo params,
                                             curandState *state,
                                             double3 *pos);

__device__ double3 thermal_pos(double temp,
                               trap_geo params,
                               curandState *state);

__host__ void cu_initialise_atom_id(int num_atoms,
                                    int *atom_id);

__global__ void g_initialise_atom_id(int num_atoms,
                                     int *atom_id);

#endif    // DISTRIBUTION_GENERATION_CUH_INCLUDED

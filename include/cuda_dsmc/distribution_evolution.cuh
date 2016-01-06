/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DISTRIBUTION_EVOLUTION_CUH_INCLUDED
#define DISTRIBUTION_EVOLUTION_CUH_INCLUDED 1

#include <cuda_runtime.h>

#include "distribution_evolution.hpp"
#include "trapping_potential.cuh"

__host__ void cu_update_atom_accelerations(int num_atoms,
                                           trap_geo params,
                                           double3 *pos,
                                           double3 *acc);

__global__ void g_update_atom_accelerations(int num_atoms,
                                            trap_geo params,
                                            double3 *pos,
                                            double3 *acc);

__device__ double3 d_update_acceleration(double3 pos,
                                         trap_geo params);

#endif    // DISTRIBUTION_EVOLUTION_CUH_INCLUDED
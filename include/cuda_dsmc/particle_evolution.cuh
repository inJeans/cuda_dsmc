/** \file
 *  \brief Predeclare functions for the evolution of distributions on
 *  the device
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef PARTILCE_EVOLUTION_CUH_INCLUDED
#define PARTILCE_EVOLUTION_CUH_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <string>

#if defined(MPI)
#include <mpi.h>
#endif

#include "cuda_dsmc/declare_physical_constants.cuh"
#include "cuda_dsmc/dsmc_utils.cuh"
#include "cuda_dsmc/vector_math.cuh"
#include "cuda_dsmc/magnetic_field.cuh"

void evolveParticleDistribution(int num_particles,
                                FieldParams params,
                                double dt,
                                cudaStream_t *streams,
                                double3 **pos,
                                double3 **vel);

__host__ void cuEvolveParticleDistribution(int num_particles,
                                           FieldParams params,
                                           double dt,
                                           cudaStream_t stream,
                                           double3 *pos,
                                           double3 *vel);

__global__ void gEvolveParticleDistribution(int num_particles,
                                            FieldParams params,
                                            double dt,
                                            double3* pos,
                                            double3* vel);

__device__ void dEvolveParticle(FieldParams params,
                                double dt,
                                double3 *pos,
                                double3 *vel);

#endif  // PARTILCE_EVOLUTION_CUH_INCLUDED

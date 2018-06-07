/** \file
 *  \brief Main code
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef DISTRIBUTION_GENERATION_CUH_INCLUDED
#define DISTRIBUTION_GENERATION_CUH_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#if defined(MPI)
#include <mpi.h>
#endif

#include "utils.cuh"
#include "cuda_dsmc/declare_physical_constants.cuh"
#include "cuda_dsmc/magnetic_field.cuh"
#include "cuda_dsmc/random_numbers.cuh"

__host__ void generateThermalPositionDistribution(int num_positions,
                                                  FieldParams params,
                                                  double temp,
                                                  curandState *states,
                                                  double3 **pos);

__host__ void cuGenerateThermalPositionDistribution(int num_positions,
                                                    FieldParams params,
                                                    double temp,
                                                    curandState *states,
                                                    double3 *pos);

__global__ void gGenerateThermalPosition(int num_positions,
                                         FieldParams params,
                                         double temp,
                                         curandState* states,
                                         double3 *pos);

__device__ double3 dGenerateThermalPosition(FieldParams params,
                                            double temp,
                                            curandState* state);

__host__ void generateThermalVelocityDistribution(int num_velocities,
                                                  double temp,
                                                  curandState* states,
                                                  double3 **vel);

__host__ void cuGenerateThermalVelocityDistribution(int num_velocities,
                                                    double temp,
                                                    curandState* states,
                                                    double3 *vel);

__global__ void gGenerateThermalVelocityDistribution(int num_velocities,
                                                     double temp,
                                                     curandState* state,
                                                     double3 *vel);

#endif  // DISTRIBUTION_GENERATION_CUH_INCLUDED

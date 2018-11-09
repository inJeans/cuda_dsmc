/** \file
 *  \brief Predeclare functions for the generations of distributions on
 *  the host
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef DISTRIBUTION_GENERATION_HPP_INCLUDED
#define DISTRIBUTION_GENERATION_HPP_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <string>

#if defined(MPI)
#include <mpi.h>
#endif

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/dsmc_utils.hpp"
#include "cuda_dsmc/magnetic_field.hpp"
#include "cuda_dsmc/random_numbers.hpp"

void generateThermalPositionDistribution(int num_positions,
                                         FieldParams params,
                                         double temp,
                                         pcg32x2_random_t* rng,
                                         double3 **pos);

void hGenerateThermalPositionDistribution(int num_positions,
                                          FieldParams params,
                                          double temp,
                                          pcg32x2_random_t* rng,
                                          double3 *pos);

double3 hGenerateThermalPosition(FieldParams params,
                                 double temp,
                                 pcg32x2_random_t* rng);

void generateThermalVelocityDistribution(int num_velocities,
                                          double temp,
                                          pcg32x2_random_t* rng,
                                          double3 **vel);

void hGenerateThermalVelocityDistribution(int num_velocities,
                                          double temp,
                                          pcg32x2_random_t* rng,
                                          double3 *vel);

#endif  // DISTRIBUTION_GENERATION_HPP_INCLUDED

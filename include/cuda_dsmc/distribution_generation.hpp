/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DISTRIBUTION_GENERATION_HPP_INCLUDED
#define DISTRIBUTION_GENERATION_HPP_INCLUDED 1

#ifdef CUDA
#include <cuda_runtime.h>
#endif

#include "random_numbers.hpp"

#ifdef CUDA
__host__ void generate_thermal_velocities(int num_atoms,
                                          double temp,
                                          curandState *state,
                                          double3 *vel);
#endif

void generate_thermal_velocities(int num_atoms,
                                 double temp,
                                 pcg64_random_t *state,
                                 double3 *vel);

double3 thermal_vel(double temp,
                    pcg64_random_t *state);

#endif  // DISTRIBUTION_GENERATION_HPP_INCLUDED

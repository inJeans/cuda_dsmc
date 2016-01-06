/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DISTRIBUTION_EVOLUTION_HPP_INCLUDED
#define DISTRIBUTION_EVOLUTION_HPP_INCLUDED 1

#include <cuda_runtime.h>

#include <g3log/g3log.hpp>

#include "trapping_potential.hpp"
#include "vector_math.cuh"

void update_atom_accelerations(int num_atoms,
                               trap_geo params,
                               double3 *pos,
                               double3 *acc);

double3 update_acceleration(double3 pos,
                            trap_geo params);

#endif  // DISTRIBUTION_EVOLUTION_HPP_INCLUDED

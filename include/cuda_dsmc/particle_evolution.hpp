/** \file
 *  \brief Predeclare functions for the evolution of distributions on
 *  the host
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef PARTILCE_EVOLUTION_HPP_INCLUDED
#define PARTILCE_EVOLUTION_HPP_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <string>

#if defined(MPI)
#include <mpi.h>
#endif

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/dsmc_utils.hpp"
#include "cuda_dsmc/vector_math.cuh"
#include "cuda_dsmc/magnetic_field.hpp"

void evolveParticleDistribution(int num_particles,
                                FieldParams params,
                                double dt,
                                double3 **pos,
                                double3 **vel);

void hEvolveParticleDistribution(int num_particles,
                                 FieldParams params,
                                 double dt,
                                 double3 *pos,
                                 double3 *vel);

void hEvolveParticle(FieldParams params,
                     double dt,
                     double3 *pos,
                     double3 *vel);

#endif  // PARTILCE_EVOLUTION_HPP_INCLUDED

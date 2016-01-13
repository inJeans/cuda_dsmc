/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DISTRIBUTION_EVOLUTION_HPP_INCLUDED
#define DISTRIBUTION_EVOLUTION_HPP_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <g3log/g3log.hpp>

#include "trapping_potential.hpp"
#include "vector_math.cuh"

void velocity_verlet_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc,
                            cublasHandle_t handle);

void sympletic_euler_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc);

void update_positions(int num_atoms,
                      double dt,
                      double3 *vel,
                      double3 *pos,
                      cublasHandle_t handle);

double3 update_atom_position(double dt,
                             double3 pos,
                             double3 vel);

void update_velocities(int num_atoms,
                       double dt,
                       double3 *acc,
                       double3 *vel,
                       cublasHandle_t handle);

double3 update_atom_velocity(double dt,
                             double3 vel,
                             double3 acc);

void update_accelerations(int num_atoms,
                          trap_geo params,
                          double3 *pos,
                          double3 *acc);

double3 update_atom_acceleration(trap_geo params,
                                 double3 pos);

#endif  // DISTRIBUTION_EVOLUTION_HPP_INCLUDED

/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef EHRENFEST_TEST_HPP_INCLUDED
#define EHRENFEST_TEST_HPP_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cub/cub.cuh>

#if defined(LOGGING)
#include <g3log/g3log.hpp>
#endif

#include "catch.hpp"

#include "distribution_generation.hpp"
#include "collisions.hpp" 
#include "distribution_evolution.hpp"

#include "utilities.hpp"
#include "test_helpers.cuh"
#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

__host__ double inst_kinetic_energy(int num_atoms,
                                    double3 *vel,
                                    double *kinetic_energy);

__host__ void cu_kinetic_energy(int num_atoms,
                                double3 *vel,
                                double *kinetic_energy);

__global__ void g_kinetic_energy(int num_atoms,
                                 double3 *vel,
                                 double *kinetic_energy);

__device__ double d_kinetic_energy(double3 vel);

__host__ double inst_potential_energy(int num_atoms,
                                      double3 *pos,
                                      trap_geo params,
                                      wavefunction *psi,
                                      double *potential_energy);

__host__ void cu_potential_energy(int num_atoms,
                                  double3 *pos,
                                  trap_geo params,
                                  wavefunction *psi,
                                  double *potential_energy);

__global__ void g_potential_energy(int num_atoms,
                                   double3 *pos,
                                   trap_geo params,
                                   wavefunction *psi,
                                   double *potential_energy);

__device__ double d_potential_energy(double3 pos,
                                     trap_geo params,
                                     wavefunction psi);

#endif  // EHRENFEST_TEST_HPP_INCLUDED
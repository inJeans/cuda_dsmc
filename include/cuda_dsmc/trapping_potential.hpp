/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef TRAPPING_POTENTIAL_HPP_INCLUDED
#define TRAPPING_POTENTIAL_HPP_INCLUDED 1

#include "trapping_potential.cuh"

__host__ double kinetic_energy(double3 vel);

__host__ double potential_energy(double3 pos,
                                 trap_geo params);

__host__ double dV_dx(double3 pos,
                      trap_geo params);

__host__ double expectation_dV_dx(trap_geo params,
                                  double3 pos,
                                  wavefunction psi);

__host__ double expectation_dV_dx(trap_geo params,
                                  double3 pos,
                                  zomplex2 psi);

__host__ double dV_dy(double3 pos,
                      trap_geo params);

__host__ double expectation_dV_dy(trap_geo params,
                                  double3 pos,
                                  wavefunction psi);

__host__ double expectation_dV_dy(trap_geo params,
                                  double3 pos,
                                  zomplex2 psi);

__host__ double dV_dz(double3 pos,
                      trap_geo params);

__host__ double expectation_dV_dz(trap_geo params,
                                  double3 pos,
                                  wavefunction psi);

__host__ double expectation_dV_dz(trap_geo params,
                                  double3 pos,
                                  zomplex2 psi);

#endif  // TRAPPING_POTENTIAL_HPP_INCLUDED

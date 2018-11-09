/** \file
 *  \brief Vector functions
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/vector_math.cuh"
#include "cuda_dsmc/magnetic_field.hpp"

#if defined(MPI)
#include <mpi.h>
#endif

void numberElementsPerParallelUnit(int unit_id,
                                   int num_units,
                                   int *num_elements);

double mean(double3 *array,
            int num_elements,
            double3 *directional_mean);

double stddev(double3 *array,
              int num_elements,
              double3 *directional_stddev);

double3 directionalKineticEnergy(double3 vel);

double kineticEnergyMean(double3 *vel,
                         int num_elements,
                         double3 *directional_energy);

double kineticEnergyStddev(double3 *vel,
                           int num_elements,
                           double3 *directional_stddev);

double3 directionalPotentialEnergy(FieldParams params, 
                                   double3 pos);

double potentialEnergyMean(double3 *pos,
                           int num_elements,
                           FieldParams params,
                           double3 *directional_energy_mean);

#endif  // UTILS_HPP_INCLUDED

/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef COLLISIONS_TEST_HPP_INCLUDED
#define COLLISIONS_EVOLUTION_TEST_HPP_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "distribution_generation.hpp"
#include "collisions.hpp" 

#include "utilities.hpp"
#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

__global__ void copy_d_grid_min(double3 *grid_min);

__global__ void copy_d_cell_length(double3 *cell_length);

__global__ void zero_elements(int num_elements,
                               double *array);

__global__ void negative_elements(int num_elements,
                                  int2 *array);

#endif  // COLLISIONS_EVOLUTION_TEST_HPP_INCLUDED
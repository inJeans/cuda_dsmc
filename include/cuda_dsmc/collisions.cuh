/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef COLLISIONS_CUH_INCLUDED
#define COLLISIONS_CUH_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <g3log/g3log.hpp>

#include "helper_cuda.h"
#include "vector_math.cuh"

__host__ void cu_initialise_grid_params(int num_atoms,
                                        cublasHandle_t cublas_handle,
                                        double3 *pos);

__global__ void copy_collision_params_to_device(double3 grid_min,
                                                double3 cell_length,
                                                int3 num_cells);

__host__ void cu_index_atoms(int num_atoms,
                             double3 *pos,
                             int *cell_id);

__global__ void g_index_atoms(int num_atoms,
                              double3 *pos,
                              int *cell_id);

__device__ int d_update_atom_cell_id(double3 pos);

__device__ int3 d_atom_cell_index(double3 pos);

__device__ int d_atom_cell_id(int3 cell_index);

#endif // COLLISIONS_CUH_INCLUDED
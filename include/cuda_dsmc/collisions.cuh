/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef COLLISIONS_CUH_INCLUDED
#define COLLISIONS_CUH_INCLUDED 1

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cublas_v2.h"
#if defined(LOGGING)
#include <g3log/g3log.hpp>
#endif

#include "helper_cuda.h"
#include "vector_math.cuh"
#include "random_numbers.cuh"

__host__ void cu_initialise_grid_params(int num_atoms,
                                        cublasHandle_t cublas_handle,
                                        double3 *pos);

__global__ void copy_collision_params_to_device(double3 grid_min,
                                                double3 cell_length,
                                                int3 num_cells,
                                                int FN);

__host__ void cu_index_atoms(int num_atoms,
                             double3 *pos,
                             int *cell_id);

__global__ void g_index_atoms(int num_atoms,
                              double3 *pos,
                              int *cell_id);

__device__ int d_update_atom_cell_id(double3 pos);

__device__ int3 d_atom_cell_index(double3 pos);

__device__ int d_atom_cell_id(int3 cell_index);

__host__ void cu_sort_atoms(int num_atoms,
                            int *cell_id,
                            int *atom_id);

__host__ void cu_find_cell_start_end(int num_atoms,
                                     int *cell_id,
                                     int2 *cell_start_end);

__global__ void g_find_cell_start_end(int num_atoms,
                                      int *cell_id,
                                      int2 *cell_start_end);

__host__ void cu_find_cell_num_atoms(int num_cells,
                                     int2 *cell_start_end,
                                     int *cell_num_atoms);

__global__ void g_find_cell_num_atoms(int num_cells,
                                      int2 *cell_start_end,
                                      int *cell_num_atoms);

__host__ void cu_scan(int num_cells,
                      int *cell_num_atoms,
                      int *cell_cumulative_num_atoms);

__host__ void cu_collide(int num_cells,
                         int *cell_id,
                         int *cell_cumulative_num_atoms,
                         double dt,
                         curandState *state,
                         double *collision_count,
                         double *collision_remainder,
                         double  *sig_vr_max,
                         double3 *vel);

__global__ void g_collide(int num_cells,
                          int *cell_id,
                          int *cell_cumulative_num_atoms,
                          double dt,
                          curandState *state,
                          double *collision_count,
                          double *collision_remainder,
                          double  *sig_vr_max,
                          double3 *vel);

__device__ int2 d_choose_colliding_atoms(int cell_num_atoms,
                                         int cell_cumulative_num_atoms,
                                         curandState *state);

__device__ int2 d_local_collision_pair(int cell_num_atoms,
                                       curandState *state);

__device__ double d_calculate_relative_velocity(double3 *vel,
                                                int2 colliding_atoms);

__device__ double3 d_random_point_on_unit_sphere(curandState *state);

#endif // COLLISIONS_CUH_INCLUDED
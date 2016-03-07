/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef COLLISIONS_HPP_INCLUDED
#define COLLISIONS_HPP_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "helper_cuda.h"

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
 
#if defined(MKL)
#include <mkl.h>
#else
#if defined(__APPLE__) && defined(__MACH__)
#include <Accelerate/Accelerate.h>
// #include <Accelerate/../Frameworks/vecLib.framework/Headers/vecLib.h>
// #include <vecLib/cblas.h>
#else
extern "C"
{
    #include <cblas.h>
}
#endif  // OS
#endif  // Parallel
#include <cmath>
#include <math.h>

#include <g3log/g3log.hpp>

#include "vector_math.cuh"
#include "random_numbers.hpp"
#include "collisions.cuh"

void initialise_grid_params(int num_atoms,
                            cublasHandle_t cublas_handle,
                            double3 *pos);

void collide_atoms(int num_atoms,
                   int num_cells,
                   double dt,
                   double3 *pos,
                   double3 *vel,
                   curandState *state,
                   double *sig_vr_max,
                   int *cell_id,
                   int *atom_id,
                   int2 *cell_start_end,
                   int *cell_num_atoms,
                   int *cell_cumulative_num_atoms,
                   double *collision_remainder,
                   int *collision_count);

void collide_atoms(int num_atoms,
                   int num_cells,
                   double dt,
                   double3 *pos,
                   double3 *vel,
                   pcg32_random_t *state,
                   double *sig_vr_max,
                   int *cell_id,
                   int *atom_id,
                   int2 *cell_start_end,
                   int *cell_num_atoms,
                   int *cell_cumulative_num_atoms,
                   double *collision_remainder,
                   int *collision_count);

void index_atoms(int num_atoms,
                 double3 *pos,
                 int *cell_id);

int update_atom_cell_id(double3 pos);

int3 atom_cell_index(double3 pos);

int atom_cell_id(int3 cell_index);

void sort_atoms(int num_atoms,
                int *cell_id,
                int *atom_id);

void count_atoms(int num_atoms,
                 int num_cells,
                 int *cell_id,
                 int2 *cell_start_end,
                 int *cell_num_atoms,
                 int *cell_cumulative_num_atoms);

void find_cell_start_end(int num_atoms,
                         int *cell_id,
                         int2 *cell_start_end);

void find_cell_num_atoms(int num_cells,
                         int2 *cell_start_end,
                         int *cell_num_atoms);

void collide(int num_cells,
             int *cell_id,
             int *cell_cumulative_num_atoms,
             double dt,
             curandState *state,
             int *collision_count,
             double *collision_remainder,
             double  *sig_vr_max,
             double3 *vel);

void collide(int num_cells,
             int *cell_id,
             int *cell_cumulative_num_atoms,
             double dt,
             pcg32_random_t *state,
             int *collision_count,
             double *collision_remainder,
             double  *sig_vr_max,
             double3 *vel);

int2 choose_colliding_atoms(int cell_num_atoms,
                            int cell_cumulative_num_atoms,
                            pcg32_random_t *state);

int2 local_collision_pair(int cell_num_atoms,
                          pcg32_random_t *state);

double calculate_relative_velocity(double3 *vel,
                                   int2 colliding_atoms);

double3 random_point_on_unit_sphere(pcg32_random_t *state);

#endif // COLLISIONS_HPP_INCLUDED

/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef COLLISIONS_HPP_INCLUDED
#define COLLISIONS_HPP_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "helper_cuda.h"
 
#if defined(MKL)
#include <mkl.h>
#else
#if defined(__APPLE__) && defined(__MACH__)
#include <Accelerate/Accelerate.h>
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

void initialise_grid_params(int num_atoms,
                            cublasHandle_t cublas_handle,
                            double3 *pos);

void collide_atoms(int num_atoms,
                   double3 *pos,
                   int *cell_id,
                   int *atom_id);

void index_atoms(int num_atoms,
                 double3 *pos,
                 int *cell_id);

int update_atom_cell_id(double3 pos);

int3 atom_cell_index(double3 pos);

int atom_cell_id(int3 cell_index);

void sort_atoms(int num_atoms,
                int *cell_id,
                int *atom_id);

#endif // COLLISIONS_HPP_INCLUDED
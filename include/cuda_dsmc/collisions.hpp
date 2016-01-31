/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef COLLISIONS_HPP_INCLUDED
#define COLLISIONS_HPP_INCLUDED 1

#include <cuda_runtime.h>
// extern "C"
// {
// 	#include <cblas.h>
// }
#include <Accelerate/Accelerate.h>
#include <math.h>

void initialise_grid_params(int num_atoms,
                            double3 *pos);

void collide_atoms(int num_atoms,
                   double3 *pos,
                   int *cell_id);

void index_atoms(int num_atoms,
                 double3 *pos,
                 int *cell_id);

int update_atom_cell_id(double3 pos);

int3 atom_cell_index(double3 pos);

int atom_cell_id(int3 cell_index);

#endif // COLLISIONS_HPP_INCLUDED
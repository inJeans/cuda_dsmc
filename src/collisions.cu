/** \file
 *  \brief Functions necessary for colliding a distribution of atoms on device
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions.cuh"

#include "declare_device_constants.cuh"

/** \fn __host__ void cu_index_atoms(int num_atoms,
 *                                   double3 *pos,
 *                                   int *cell_id) 
 *  \brief Calls the `__global__` function to update an `int` device array with
 *  cell_ids based on the atoms position and the maximum cell width.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *pos Pointer to a `double3` device array of length
 *  `num_atoms` containing the positions.
 *  \param *cell_id Pointer to an output `int` device array of length
 *  `num_atoms` containing the cell_ids.
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_index_atoms(int num_atoms,
                             double3 *pos,
                             int *cell_id) {
LOGF(DEBUG, "\nCalculating optimal launch configuration for the atom "
                "indexing kernel.\n");
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_index_atoms,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
    g_index_atoms<<<grid_size,
                    block_size>>>
                   (num_atoms,
                    pos,
                    cell_id);

    return;
}

/** \fn __global__ void g_index_atoms(int num_atoms,
 *                                    double3 *pos,
 *                                    int *cell_id) 
 *  \brief `__global__` function to update an `int` device array with
 *  cell_ids based on the atoms position and the maximum cell width.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *pos Pointer to a `double3` device array of length
 *  `num_atoms` containing the positions.
 *  \param *cell_id Pointer to an output `int` device array of length
 *  `num_atoms` containing the cell_ids.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_index_atoms(int num_atoms,
                              double3 *pos,
                              int *cell_id) {
    for (int atom = 0; atom < num_atoms; ++atom) {
        cell_id[atom] = d_update_atom_cell_id(pos[atom]);
    }

    return;
}

/** \fn __device__ int d_update_atom_cell_id(double3 pos) 
 *  \brief Calls the function to calculate the cell ID of an atom based on its
 *  current position. Cell IDs are counted from the negative end of each
 *  cartesian direction, first along `x`, then along `y` and finally along `z`.
 *  \param pos The position of the atom.
 *  \exception not yet.
 *  \return cell_id An integer containing the cell ID of the atom.
*/

__device__ int d_update_atom_cell_id(double3 pos) {
    int cell_id = 0;

    int3 cell_index = d_atom_cell_index(pos);
    cell_id = d_atom_cell_id(cell_index);

    return cell_id;
}

/** \fn __device__ int3 d_atom_cell_index(double3 pos) 
 *  \brief Calls the function to calculate the individual cell indicies for each
 *  cartesian direction based on an atoms current position.
 *  \param pos The position of the atom.
 *  \exception not yet.
 *  \return cell_index An `int3` containing the individual cell indicies for
 *  each cartesian direction.
*/

__device__ int3 d_atom_cell_index(double3 pos) {
    int3 cell_index = make_int3(0, 0, 0);

    // NOTE: Computer scientists may have a problem with this typecast since,
    //       integers cannot store the same maximum number as a float can.
    //       So if we anticipate having more than 2^31 cells, then we need
    //       to do something smarter here.
    cell_index = type_cast_int3(floor((pos - d_grid_min) / d_cell_length));

    return cell_index;
}

/** \fn __device__ int d_atom_cell_id(int3 cell_index) 
 *  \brief Calls the function to combine the individual cell indicies for each
 *  cartesian direction into the singal global `cell_id`.
 *  \param cell_index The cartesian cell indices of the atom.
 *  \exception not yet.
 *  \return cell_index An `int` containing the global `cell_id`.
*/

__device__ int d_atom_cell_id(int3 cell_index) {
    int cell_id = 0;

    if (cell_index.x > -1 && cell_index.x < d_num_cells.x &&
        cell_index.y > -1 && cell_index.y < d_num_cells.y &&
        cell_index.z > -1 && cell_index.z < d_num_cells.z) {
        cell_id = cell_index.z*d_num_cells.x*d_num_cells.y +
                  cell_index.y*d_num_cells.x +
                  cell_index.x;
    } else {
        cell_id = d_num_cells.x*d_num_cells.y*d_num_cells.z;
    }

    return cell_id;
}

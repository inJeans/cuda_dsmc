/** \file
 *  \brief Functions necessary for colliding a distribution of atoms
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions.hpp"
#if defined(CUDA)
#include "collisions.cuh"
#endif

#include "declare_host_constants.hpp"
#if defined(CUDA)
#include "declare_device_constants.cuh"
#endif

void initialise_grid_params(int num_atoms,
                            cublasHandle_t cublas_handle,
                            double3 *pos) {
    LOGF(INFO, "\nInitialising grid parameters.\n");
    int3 max_id = make_int3(0, 0, 0);
#if defined(CUDA)
    cu_initialise_grid_params(num_atoms,
                              cublas_handle,
                              pos);
#else
    LOGF(DEBUG, "\nLaunching BLAS idamax to find max x position.\n");
    max_id.x = cblas_idamax(num_atoms,
                            reinterpret_cast<double *>(pos)+0,
                            3);
    LOGF(DEBUG, "\nLaunching BLAS idamax to find max y position.\n");
    max_id.y = cblas_idamax(num_atoms,
                            reinterpret_cast<double *>(pos)+1,
                            3);
    LOGF(DEBUG, "\nLaunching BLAS idamax to find max z position.\n");
    max_id.z = cblas_idamax(num_atoms,
                            reinterpret_cast<double *>(pos)+2,
                            3);
    LOGF(DEBUG, "\nThe ids of the max positions are max_id = {%i, %i, %i\n",
         max_id.x, max_id.y, max_id.z);
    grid_min.x = -1.0*std::abs(pos[max_id.x].x);
    grid_min.y = -1.0*std::abs(pos[max_id.y].y);
    grid_min.z = -1.0*std::abs(pos[max_id.z].z);
    LOGF(DEBUG, "\nThe minimum grid points are grid_min = {%f, %f, %f}\n",
         grid_min.x, grid_min.y, grid_min.z);
#endif

    // Set the grid_max = -grid_min, so that the width of the grid would be
    // 2*abs(grid_min) or -2.0 * grid_min.
    cell_length = -2.0 * grid_min / num_cells;
    LOGF(DEBUG, "\nThe cell widths are cell_length = {%f, %f, %f}\n",
         cell_length.x, cell_length.y, cell_length.z);

    return;
}

/** \fn void collide_atoms(int num_atoms,
 *                         double3 *pos,
 *                         int *cell_id,
 *                         int *atom_id) 
 *  \brief Calls the function to simulate collisions between the atoms of the
 *  thermal gas. The collisions rates should match those predicted by the
 *  kinetic theory of gases. Collisions are approximated to be `s`-wave.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \param *cell_id Pointer to an output `int` host or device array of length
 *  `num_atoms` containing the cell_ids.
 *  \param *cell_id Pointer to an input `int` host or device array of length
 *  `num_atoms` containing the atom_ids.
 *  \exception not yet.
 *  \return void
*/

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

void collide_atoms(int num_atoms,
                   double3 *pos,
                   int *cell_id,
                   int *atom_id,
                   int2 *cell_start_end) {
    // Index atoms
    index_atoms(num_atoms,
                pos,
                cell_id);
    // Sort atoms
    sort_atoms(num_atoms,
               cell_id,
               atom_id);
    // Count attoms
    find_cell_start_end(num_atoms,
                        cell_id,
                        cell_start_end);
    // Collide atoms
    return;
}

/****************************************************************************
 * INDEXING                                                                 *
 ****************************************************************************/

/** \fn void index_atoms(int num_atoms,
 *                       double3 *pos,
 *                       int *cell_id) 
 *  \brief Calls the function to update an `int` host or device array with
 *  cell_ids based on the atoms position and the maximum cell width.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \param *cell_id Pointer to an output `int` host or device array of length
 *  `num_atoms` containing the cell_ids.
 *  \exception not yet.
 *  \return void
*/

void index_atoms(int num_atoms,
                 double3 *pos,
                 int *cell_id) {
#ifdef CUDA
    cu_index_atoms(num_atoms,
                   pos,
                   cell_id);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        cell_id[atom] = update_atom_cell_id(pos[atom]);
    }
#endif

    return;
}

/** \fn int update_atom_cell_id(double3 pos) 
 *  \brief Calls the function to calculate the cell ID of an atom based on its
 *  current position. Cell IDs are counted from the negative end of each
 *  cartesian direction, first along `x`, then along `y` and finally along `z`.
 *  \param pos The position of the atom.
 *  \exception not yet.
 *  \return cell_id An integer containing the cell ID of the atom.
*/

int update_atom_cell_id(double3 pos) {
    int cell_id = 0;

    int3 cell_index = atom_cell_index(pos);
    cell_id = atom_cell_id(cell_index);

    return cell_id;
}

/** \fn int3 atom_cell_index(double3 pos) 
 *  \brief Calls the function to calculate the individual cell indicies for each
 *  cartesian direction based on an atoms current position.
 *  \param pos The position of the atom.
 *  \exception not yet.
 *  \return cell_index An `int3` containing the individual cell indicies for
 *  each cartesian direction.
*/

int3 atom_cell_index(double3 pos) {
    int3 cell_index = make_int3(0, 0, 0);

    // NOTE: Computer scientists may have a problem with this typecast since,
    //       integers cannot store the same maximum number as a float can.
    //       So if we anticipate having more than 2^31 cells, then we need
    //       to do something smarter here.
    cell_index = type_cast_int3(floor((pos - grid_min) / cell_length));

    return cell_index;
}

/** \fn int atom_cell_id(int3 cell_index) 
 *  \brief Calls the function to combine the individual cell indicies for each
 *  cartesian direction into the singal global `cell_id`.
 *  \param cell_index The cartesian cell indices of the atom.
 *  \exception not yet.
 *  \return cell_index An `int` containing the global `cell_id`.
*/

int atom_cell_id(int3 cell_index) {
    int cell_id = 0;

    if (cell_index.x > -1 && cell_index.x < num_cells.x &&
        cell_index.y > -1 && cell_index.y < num_cells.y &&
        cell_index.z > -1 && cell_index.z < num_cells.z) {
        cell_id = cell_index.z*num_cells.x*num_cells.y +
                  cell_index.y*num_cells.x +
                  cell_index.x;
    } else {
        cell_id = num_cells.x*num_cells.y*num_cells.z;
    }

    return cell_id;
}

/****************************************************************************
 * SORTING                                                                  *
 ****************************************************************************/

/** \fn void sort_atoms(int num_atoms,
 *                      int *cell_id,
 *                      int *atom_id) 
 *  \brief Calls the function to sort an `int` host or device array with
 *  atom_ids based on the cell_ids of the atoms.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *cell_id Pointer to an input/output `int` host or device array of
 *  length `num_atoms` containing the cell_ids.
 *  \param *atom_id Pointer to an input/output `int` host or device array of
 *  length `num_atoms` containing the atom_ids.
 *  \exception not yet.
 *  \return void
*/

void sort_atoms(int num_atoms,
                int *cell_id,
                int *atom_id) {
#if defined(CUDA)
    cu_sort_atoms(num_atoms,
                  cell_id,
                  atom_id);
#else
    thrust::sort_by_key(thrust::host,
                        cell_id,
                        cell_id + num_atoms,
                        atom_id);
#endif

    return;
}

/****************************************************************************
 * COUNTING                                                                 *
 ****************************************************************************/

void find_cell_start_end(int num_atoms,
                         int *cell_id,
                         int2 *cell_start_end) {
#if defined(CUDA)
    cu_find_cell_start_end(num_atoms,
                           cell_id,
                           cell_start_end);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        int l_cell_id = cell_id[atom];
        // Find the beginning of the cell
        if (atom == 0) {
            cell_start_end[l_cell_id].x = 0;
        } else if (l_cell_id != cell_id[atom-1]) {
            cell_start_end[l_cell_id].x = atom;
        }

        // Find the end of the cell
        if (atom == num_atoms - 1) {
            cell_start_end[l_cell_id].y = num_atoms-1;
        } else if (l_cell_id != cell_id[atom+1]) {
            cell_start_end[l_cell_id].y = atom;
        }
    }
#endif

    return;
}

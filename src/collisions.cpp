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
                            double3 *pos) {
    int3 max_id = make_int3(0, 0, 0);
    max_id.x = cblas_idamax(num_atoms,
                            reinterpret_cast<double *>(pos)+0,
                            3);
    max_id.y = cblas_idamax(num_atoms,
                            reinterpret_cast<double *>(pos)+1,
                            3);
    max_id.z = cblas_idamax(num_atoms,
                            reinterpret_cast<double *>(pos)+2,
                            3);

    grid_min.x = -1.0*std::abs(pos[max_id.x].x);
    grid_min.y = -1.0*std::abs(pos[max_id.y].y);
    grid_min.z = -1.0*std::abs(pos[max_id.z].z);

    // Set the grid_max = -grid_min, so that the width of the grid would be
    // 2*abs(grid_min) or -2.0 * grid_min.
    cell_length = -2.0 * grid_min / num_cells;

#if defined(CUDA)
    d_grid_min = grid_min;
    d_cell_length = cell_length;
#endif
    return;
}

/** \fn void collide_atoms(int num_atoms,
 *                       double3 *pos,
 *                       int *cell_id) 
 *  \brief Calls the function to simulate collisions between the atoms of the
 *  thermal gas. The collisions rates should match those predicted by the
 *  kinetic theory of gases. Collisions are approximated to be `s`-wave.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \param *cell_id Pointer to an output `int` host or device array of length
 *  `num_atoms` containing the cell_ids.
 *  \exception not yet.
 *  \return void
*/

void collide_atoms(int num_atoms,
                   double3 *pos,
                   int *cell_id) {
    // Index atoms
    index_atoms(num_atoms,
                pos,
                cell_id);
    // Sort atoms
    // Bin atoms
    // Collide atoms
    return;
}

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

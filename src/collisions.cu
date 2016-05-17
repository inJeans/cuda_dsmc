/** \file
 *  \brief Functions necessary for colliding a distribution of atoms on device
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions.cuh"
#include <cub/cub.cuh> // Need to keep this include out of the main header
                       // as regular c compilers might not like it

#include "declare_host_constants.hpp"
#include "declare_device_constants.cuh"

__host__ void cu_initialise_grid_params(int num_atoms,
                                        cublasHandle_t cublas_handle,
                                        double3 *pos) {
    int3 max_id = make_int3(0, 0, 0);
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunching cuBLAS idamax to find max x position.\n");
#endif
    checkCudaErrors(cublasIdamax(cublas_handle,
                                 num_atoms,
                                 reinterpret_cast<double *>(pos)+0,
                                 3,
                                 &max_id.x));
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunching cuBLAS idamax to find max y position.\n");
#endif
    checkCudaErrors(cublasIdamax(cublas_handle,
                                 num_atoms,
                                 reinterpret_cast<double *>(pos)+1,
                                 3,
                                 &max_id.y));
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunching cuBLAS idamax to find max z position.\n");
#endif
    checkCudaErrors(cublasIdamax(cublas_handle,
                                 num_atoms,
                                 reinterpret_cast<double *>(pos)+2,
                                 3,
                                 &max_id.z));
    // cuBLAS returns indices with FORTRAN 1-based indexing.
    max_id = max_id - 1;
#if defined(LOGGING)
    LOGF(DEBUG, "\nThe ids of the max positions are max_id = {%i, %i, %i}\n",
         max_id.x, max_id.y, max_id.z);
#endif
    double3 *h_pos;
    h_pos = reinterpret_cast<double3*>(calloc(num_atoms,
                                              sizeof(double3)));
    checkCudaErrors(cudaMemcpy(h_pos,
                               pos,
                               num_atoms*sizeof(double3),
                               cudaMemcpyDeviceToHost));
    grid_min.x = -1.0*std::abs(h_pos[max_id.x].x);
    grid_min.y = -1.0*std::abs(h_pos[max_id.y].y);
    grid_min.z = -1.0*std::abs(h_pos[max_id.z].z);
    free(h_pos);
#if defined(LOGGING)
    LOGF(DEBUG, "\nThe minimum grid points are grid_min = {%f, %f, %f}\n",
         grid_min.x, grid_min.y, grid_min.z);
#endif
    // Set the grid_max = -grid_min, so that the width of the grid would be
    // 2*abs(grid_min) or -2.0 * grid_min.
    cell_length = -2.0 * grid_min / k_num_cells;
    
    copy_collision_params_to_device<<<1, 1>>>(grid_min,
                                              cell_length,
                                              k_num_cells,
                                              FN);
#if defined(LOGGING)
    LOGF(DEBUG, "\nThe minimum grid points on the device are d_grid_min = {%f, %f, %f}\n",
         grid_min.x, grid_min.y, grid_min.z);
    LOGF(DEBUG, "\nThe cell widths on the device are d_cell_length = {%f, %f, %f}\n",
         cell_length.x, cell_length.y, cell_length.z);
#endif
  return;
}

__global__ void copy_collision_params_to_device(double3 grid_min,
                                                double3 cell_length,
                                                int3 num_cells,
                                                int FN) {
    d_grid_min = grid_min;
    d_cell_length = cell_length;
    d_cell_volume = d_cell_length.x * d_cell_length.y * d_cell_length.z;
    d_num_cells = num_cells;

    d_cross_section = 8. * d_pi * d_a * d_a;

    d_FN = FN;

    return;
}

/****************************************************************************
 * INDEXING                                                                 *
 ****************************************************************************/

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
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the atom "
                "indexing kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_index_atoms,
                                       0,
                                       num_atoms);
    if (block_size < 1) block_size = 1;
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif
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

/****************************************************************************
 * SORTING                                                                  *
 ****************************************************************************/

/** \fn __host__ void cu_sort_atoms(int num_atoms,
 *                                  int *cell_id,
 *                                  int *atom_id) 
 *  \brief Calls the function to sort an `int` device array with atom_ids 
 *  based on the cell_ids of the atoms.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *cell_id Pointer to an input/output `int` device array of length
 *  `num_atoms` containing the cell_ids.
 *  \param *atom_id Pointer to an input/output `int` device array of length
 *  `num_atoms` containing the atom_ids.
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_sort_atoms(int num_atoms,
                            int *cell_id,
                            int *atom_id) {
    int  *d_cell_id_out;
    int  *d_atom_id_out;

    checkCudaErrors(cudaMalloc(&d_cell_id_out,
                               num_atoms*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_atom_id_out,
                               num_atoms*sizeof(int)));

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    CubDebug(cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             cell_id,
                                             d_cell_id_out,
                                             atom_id,
                                             d_atom_id_out,
                                             num_atoms));
    // Allocate temporary storage
    checkCudaErrors(cudaMalloc(&d_temp_storage,
                               temp_storage_bytes));
    // Run sorting operation
    CubDebug(cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             cell_id,
                                             d_cell_id_out,
                                             atom_id,
                                             d_atom_id_out,
                                             num_atoms));
    // Copy sorted arrays back to original memory
    checkCudaErrors(cudaMemcpy(atom_id,
                               d_atom_id_out,
                               num_atoms*sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cell_id,
                               d_cell_id_out,
                               num_atoms*sizeof(int),
                               cudaMemcpyDeviceToHost));

    cudaFree(d_cell_id_out);
    cudaFree(d_atom_id_out);
    cudaFree(d_temp_storage);
    return;
}

/****************************************************************************
 * COUNTING                                                                 *
 ****************************************************************************/

__host__ void cu_find_cell_start_end(int num_atoms,
                                     int *cell_id,
                                     int2 *cell_start_end) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the cell "
                "start/end kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_find_cell_start_end,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif
    g_find_cell_start_end<<<grid_size,
                            block_size>>>
                           (num_atoms,
                            cell_id,
                            cell_start_end);

    return;
}

__global__ void g_find_cell_start_end(int num_atoms,
                                      int *cell_id,
                                      int2 *cell_start_end) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
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

    return;
}

__host__ void cu_find_cell_num_atoms(int num_cells,
                                     int2 *cell_start_end,
                                     int *cell_num_atoms) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the cell "
                "atom counting kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_find_cell_num_atoms,
                                       0,
                                       num_cells+1);
    grid_size = (num_cells+1 + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif
    g_find_cell_num_atoms<<<grid_size,
                            block_size>>>
                           (num_cells,
                            cell_start_end,
                            cell_num_atoms);

    return;
}

__global__ void g_find_cell_num_atoms(int num_cells,
                                      int2 *cell_start_end,
                                      int *cell_num_atoms) {
    for (int cell = blockIdx.x * blockDim.x + threadIdx.x;
         cell < num_cells+1;
         cell += blockDim.x * gridDim.x) {
        if (cell_start_end[cell].x == -1)
            cell_num_atoms[cell] = 0;
        else
            cell_num_atoms[cell] = cell_start_end[cell].y -
                                   cell_start_end[cell].x + 1;
    }

    return;
}

__host__ void cu_scan(int num_cells,
                      int *cell_num_atoms,
                      int *cell_cumulative_num_atoms) {
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebug(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                               temp_storage_bytes,
                                               cell_num_atoms,
                                               cell_cumulative_num_atoms,
                                               num_cells+1));
    // Allocate temporary storage
    checkCudaErrors(cudaMalloc(&d_temp_storage,
                               temp_storage_bytes));
    // Run exclusive prefix sum
    CubDebug(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                               temp_storage_bytes,
                                               cell_num_atoms,
                                               cell_cumulative_num_atoms,
                                               num_cells+1));
    int num_atoms = 0;
    checkCudaErrors(&num_atoms,
                    &cell_cumulative_num_atoms[num_cells],
                    sizeof(int),
                    cudaMemcpyDeviceToHost));
    print("Number of atoms = %i\n", num_atoms);

    cudaFree(d_temp_storage);
    return;
}

/****************************************************************************
 * COLLIDING                                                                *
 ****************************************************************************/

__host__ void cu_collide(int num_cells,
                         int *cell_id,
                         int *cell_cumulative_num_atoms,
                         double dt,
                         curandState *state,
                         int *collision_count,
                         double *collision_remainder,
                         double  *sig_vr_max,
                         double3 *vel) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the atom "
                "collision kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_collide,
                                       0,
                                       num_cells);
    if (block_size < 1) block_size = 1;
    grid_size = (num_cells + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n", grid_size, block_size);
#endif
    g_collide<<<grid_size,
                block_size>>>
             (num_cells,
              cell_id,
              cell_cumulative_num_atoms,
              dt,
              state,
              collision_count,
              collision_remainder,
              sig_vr_max,
              vel);

    return;
}

__global__ void g_collide(int num_cells,
                          int *cell_id,
                          int *cell_cumulative_num_atoms,
                          double dt,
                          curandState *state,
                          int *collision_count,
                          double *collision_remainder,
                          double  *sig_vr_max,
                          double3 *vel) {
    for (int cell = blockIdx.x * blockDim.x + threadIdx.x;
         cell < num_cells;
         cell += blockDim.x * gridDim.x) {
        int cell_num_atoms = cell_cumulative_num_atoms[cell+1] -
                             cell_cumulative_num_atoms[cell];

        double l_sig_vr_max = sig_vr_max[cell];
        curandState l_state = state[cell];

        double f_num_collision_pairs = 0.5 * cell_num_atoms * cell_num_atoms *
                                       d_FN * l_sig_vr_max * dt / d_cell_volume +
                                       collision_remainder[cell];
        int num_collision_pairs = floor(f_num_collision_pairs);
        collision_remainder[cell] = f_num_collision_pairs - num_collision_pairs;
        
        if (cell_num_atoms > 2) {
            double3 vel_cm, new_vel, point_on_sphere;

            double mag_rel_vel;
            double prob_collision;

            for (int l_collision = 0;
                 l_collision < num_collision_pairs;
                 l_collision++ ) {
                int2 colliding_atoms = make_int2(0, 0);

                colliding_atoms = d_choose_colliding_atoms(cell_num_atoms,
                                                           cell_cumulative_num_atoms[cell],
                                                           &l_state);

                mag_rel_vel = d_calculate_relative_velocity(vel,
                                                            colliding_atoms);

                // Check if this is the more probable than current
                // most probable.
                if (mag_rel_vel*d_cross_section > l_sig_vr_max) {
                    l_sig_vr_max = mag_rel_vel * d_cross_section;
                }

                prob_collision = mag_rel_vel*d_cross_section / l_sig_vr_max;
                // printf("cell[%i]: #-col = %i, prob-coll = %f\n", cell, num_collision_pairs, prob_collision);

                // Collide with the collision probability.
                if (prob_collision > curand_uniform_double(&l_state)) {
                    // Find centre of mass velocities.
                    vel_cm = 0.5*(vel[colliding_atoms.x] +
                                  vel[colliding_atoms.y]);

                    // Generate a random velocity on the unit sphere.
                    point_on_sphere = d_random_point_on_unit_sphere(&l_state);
                    new_vel = mag_rel_vel * point_on_sphere;

                    vel[colliding_atoms.x] = vel_cm - 0.5 * new_vel;
                    vel[colliding_atoms.y] = vel_cm + 0.5 * new_vel;

                    //            atomicAdd( &collisionCount[cell], d_alpha );
                    // collision_count[cell] += d_FN;
                    collision_count[cell] = cell_num_atoms;
                }
            }
        }
        state[cell] = l_state;
        sig_vr_max[cell] = l_sig_vr_max;
    }
    return;
}

__device__ int2 d_choose_colliding_atoms(int cell_num_atoms,
                                         int cell_cumulative_num_atoms,
                                         curandState *state) {
    int2 colliding_atoms = make_int2(0, 0);

    if (cell_num_atoms == 2) {
        colliding_atoms.x = cell_cumulative_num_atoms + 0;
        colliding_atoms.y = cell_cumulative_num_atoms + 1;
    } else {
        colliding_atoms = cell_cumulative_num_atoms +
                          d_local_collision_pair(cell_num_atoms,
                                                 state);
    }

    return colliding_atoms;
}

__device__ int2 d_local_collision_pair(int cell_num_atoms,
                                       curandState *state) {
    int2 local_pair = make_int2(0, 0);

    // Randomly choose particles in this cell to collide.
    while (local_pair.x == local_pair.y) {
        local_pair.x = int(floor(curand_uniform(&state[0]) *
                                      (cell_num_atoms-1)));
        local_pair.y = int(floor(curand_uniform(&state[0]) *
                                      (cell_num_atoms-1)));
    }

    return local_pair;
}

__device__ double d_calculate_relative_velocity(double3 *vel,
                                                int2 colliding_atoms) {
    double3 vel_rel = vel[colliding_atoms.x] - vel[colliding_atoms.y];
    double mag_vel_rel = norm(vel_rel);

    return mag_vel_rel;
}

__device__ double3 d_random_point_on_unit_sphere(curandState *state) {
    double3 normal_point = d_gaussian_point(0,
                                            1,
                                            state);

    double3 point_on_sphere = normal_point / norm(normal_point);

    return point_on_sphere;
}

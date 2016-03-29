/** \file
 *  \brief Unit tests for the collisions file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions_test.cuh"

double fractional_tol = 0.05; 

SCENARIO("[DEVICE] Initialise grid parameters", "[d-initgrid]") {
    GIVEN("A device array of 10 known positions, in a grid with num_cells = {2,3,4}.") {
        double3 pos[10];
        pos[0] = make_double3(0., 0., 0.);
        pos[1] = make_double3(-1., -1., -1.);
        pos[2] = make_double3(1., 1., 1.);
        pos[3] = make_double3(-3., 0., 1.);
        pos[4] = make_double3(10., -3., 4.);
        pos[5] = make_double3(2., 9., -6.);
        pos[6] = make_double3(-8., 15., 7.);
        pos[7] = make_double3(-2., -8., 10.);
        pos[8] = make_double3(1., -2., -10.);
        pos[9] = make_double3(0., 2., 0.);

        double3 *d_pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pos),
                                   10*sizeof(double3)));
        checkCudaErrors(cudaMemcpy(d_pos,
                                   pos,
                                   10*sizeof(double3),
                                   cudaMemcpyHostToDevice));

        k_num_cells = make_int3(2, 3, 4);

        WHEN("The initialise_grid_params function is called") {
            cublasHandle_t cublas_handle;
            checkCudaErrors(cublasCreate(&cublas_handle));
            initialise_grid_params(10,
                                   cublas_handle,
                                   d_pos);

            THEN("Then the device global grid_min = {-10., -15., -10.} ") {
                // Cannot memCpy from constant memory. Need to use a kernel to
                // copy into global memory first.
                double3 *grid_min;
                checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&grid_min),
                                           sizeof(double3)));
                copy_d_grid_min<<<1, 1>>>(grid_min);
                double3 t_grid_min = make_double3(0., 0., 0.);
                checkCudaErrors(cudaMemcpy(&t_grid_min,
                                           grid_min,
                                           sizeof(double3),
                                           cudaMemcpyDeviceToHost));
                REQUIRE(t_grid_min.x == -10.);
                REQUIRE(t_grid_min.y == -15.);
                REQUIRE(t_grid_min.z == -10.);

                cudaFree(grid_min);
            }
            THEN("Then the device global cell_length = {10., 10., 5.} ") {
                // Cannot memCpy from constant memory. Need to use a kernel to
                // copy into global memory first.
                double3 *cell_length;
                checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_length),
                                           sizeof(double3)));
                copy_d_cell_length<<<1, 1>>>(cell_length);
                double3 t_cell_length = make_double3(0., 0., 0.);
                checkCudaErrors(cudaMemcpy(&t_cell_length,
                                           cell_length,
                                           sizeof(double3),
                                           cudaMemcpyDeviceToHost));
                REQUIRE(t_cell_length.x == 10.);
                REQUIRE(t_cell_length.y == 10.);
                REQUIRE(t_cell_length.z == 5.);

                cudaFree(cell_length);
            }

            checkCudaErrors(cublasDestroy(cublas_handle));
        }

        cudaFree(d_pos);
    }
}

SCENARIO("[DEVICE] Index atoms", "[d-index]") {
    GIVEN("An array of 10 known positions, in a grid with num_cells = {2,3,4}.") {
        double3 pos[10];
        pos[0] = make_double3(0., 0., 0.);
        pos[1] = make_double3(-1., -1., -1.);
        pos[2] = make_double3(1., 1., 1.);
        pos[3] = make_double3(-3., 0., 1.);
        pos[4] = make_double3(10., -3., 4.);
        pos[5] = make_double3(2., 9., -6.);
        pos[6] = make_double3(-8., 15., 7.);
        pos[7] = make_double3(-2., -8., 10.);
        pos[8] = make_double3(1., -2., -10.);
        pos[9] = make_double3(0., 2., 0.);
        double3 *d_pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pos),
                                   10*sizeof(double3)));
        checkCudaErrors(cudaMemcpy(d_pos,
                                   pos,
                                   10*sizeof(double3),
                                   cudaMemcpyHostToDevice));

        k_num_cells = make_int3(2, 3, 4);

        cublasHandle_t cublas_handle;
        checkCudaErrors(cublasCreate(&cublas_handle));
        initialise_grid_params(10,
                               cublas_handle,
                               d_pos);

        WHEN("The index_atoms function is called") {
            int *d_cell_id;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_id),
                                       10*sizeof(int)));
            checkCudaErrors(cudaMemset(d_cell_id,
                                       0,
                                       10));
            index_atoms(10,
                        d_pos,
                        d_cell_id);

            int test_cell_id[10] = {0};
            checkCudaErrors(cudaMemcpy(test_cell_id,
                                       d_cell_id,
                                       10*sizeof(int),
                                       cudaMemcpyDeviceToHost));

            THEN("Then the global cell_id = {15, 8, 15, 14, 24, 5, 24, 24, 3, 15} ") {
                REQUIRE(test_cell_id[0] == 15);
                REQUIRE(test_cell_id[1] == 8);
                REQUIRE(test_cell_id[2] == 15);
                REQUIRE(test_cell_id[3] == 14);
                REQUIRE(test_cell_id[4] == 24);
                REQUIRE(test_cell_id[5] == 5);
                REQUIRE(test_cell_id[6] == 24);
                REQUIRE(test_cell_id[7] == 24);
                REQUIRE(test_cell_id[8] == 3);
                REQUIRE(test_cell_id[9] == 15);
            }

            cudaFree(d_cell_id);
        }
        checkCudaErrors(cublasDestroy(cublas_handle));
        cudaFree(d_pos);
    }
}

SCENARIO("[DEVICE] Sort atoms", "[d-sort]") {
    GIVEN("An array of 10 known indices, with an associated atom_id array") {
        int atom_id[10] = {0,  1, 2,  3,  4,  5, 6,  7,  8, 9};
        int cell_id[10] = {15, 8, 15, 14, 24, 5, 24, 24, 3, 15};

        int *d_atom_id;
        int *d_cell_id;

        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_atom_id),
                                   10*sizeof(int)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_id),
                                   10*sizeof(int)));

        checkCudaErrors(cudaMemcpy(d_atom_id,
                                   atom_id,
                                   10*sizeof(int),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_cell_id,
                                   cell_id,
                                   10*sizeof(int),
                                   cudaMemcpyHostToDevice));


        WHEN("The sort_atoms function is called") {
            sort_atoms(10,
                       d_cell_id,
                       d_atom_id);

            checkCudaErrors(cudaMemcpy(atom_id,
                                       d_atom_id,
                                       10*sizeof(int),
                                       cudaMemcpyDeviceToHost));

            THEN("Then the global atom_id = {8, 5, 1, 3, 0, 2, 9, 4, 6, 7} ") {
                REQUIRE(atom_id[0] == 8);
                REQUIRE(atom_id[1] == 5);
                REQUIRE(atom_id[2] == 1);
                REQUIRE(atom_id[3] == 3);
                REQUIRE(atom_id[4] == 0);
                REQUIRE(atom_id[5] == 2);
                REQUIRE(atom_id[6] == 9);
                REQUIRE(atom_id[7] == 4);
                REQUIRE(atom_id[8] == 6);
                REQUIRE(atom_id[9] == 7);
            }
        }

        cudaFree(d_atom_id);
        cudaFree(d_cell_id);
    }
}

SCENARIO("[DEVICE] Count atoms", "[d-count]") {
    GIVEN("An array of 10 sorted cell_ids with num_cells = 8.") {
        int num_atoms = 10;
        int num_cells = 8;

        int cell_id[10] = {0, 2, 4, 5, 6, 6, 6, 8, 8, 8};
        int *d_cell_id;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_id),
                                   num_atoms*sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_cell_id,
                                   cell_id,
                                   num_atoms*sizeof(int),
                                   cudaMemcpyHostToDevice));

        WHEN("The sort_atoms function is called") {
            int *d_cell_num_atoms;
            int *d_cell_cumulative_num_atoms;

            int2 *d_cell_start_end;
            
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_num_atoms),
                                       (num_cells+1)*sizeof(int)));
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_cumulative_num_atoms),
                                       (num_cells+1)*sizeof(int)));
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_start_end),
                                       (num_cells+1)*sizeof(int2)));

            // checkCudaErrors(cudaMemset(d_cell_start_end,
            //                            -1,
            //                            num_atoms*sizeof(int2)));
            negative_elements<<<num_cells+1,1>>>(num_cells+1,
                                                 d_cell_start_end);

            count_atoms(num_atoms,
                        num_cells,
                        d_cell_id,
                        d_cell_start_end,
                        d_cell_num_atoms,
                        d_cell_cumulative_num_atoms);

            int t_cell_num_atoms[9];
            int t_cell_cumulative_num_atoms[9];
            int2 t_cell_start_end[9];

            checkCudaErrors(cudaMemcpy(t_cell_num_atoms,
                                       d_cell_num_atoms,
                                       (num_cells+1)*sizeof(int),
                                       cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(t_cell_cumulative_num_atoms,
                                       d_cell_cumulative_num_atoms,
                                       (num_cells+1)*sizeof(int),
                                       cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(t_cell_start_end,
                                       d_cell_start_end,
                                       (num_cells+1)*sizeof(int2),
                                       cudaMemcpyDeviceToHost));

            cudaFree(d_cell_num_atoms);
            cudaFree(d_cell_cumulative_num_atoms);
            cudaFree(d_cell_start_end);

            THEN("Then the global cell_start_end = {{0, 0}, {-1, -1}, {1, 1}, {-1, -1}, {2, 2}, {3, 3}, {4, 6}, {7, 9}}") {
                REQUIRE(t_cell_start_end[0] == make_int2(0, 0));
                REQUIRE(t_cell_start_end[1] == make_int2(-1, -1));
                REQUIRE(t_cell_start_end[2] == make_int2(1, 1));
                REQUIRE(t_cell_start_end[3] == make_int2(-1, -1));
                REQUIRE(t_cell_start_end[4] == make_int2(2, 2));
                REQUIRE(t_cell_start_end[5] == make_int2(3, 3));
                REQUIRE(t_cell_start_end[6] == make_int2(4, 6));
                REQUIRE(t_cell_start_end[7] == make_int2(-1, -1));
                REQUIRE(t_cell_start_end[8] == make_int2(7, 9));
            }

            THEN("Then the global cell_num_atoms = {1, 0, 1, 0, 1, 1, 3, 0, 3}") {
                REQUIRE(t_cell_num_atoms[0] == 1);
                REQUIRE(t_cell_num_atoms[1] == 0);
                REQUIRE(t_cell_num_atoms[2] == 1);
                REQUIRE(t_cell_num_atoms[3] == 0);
                REQUIRE(t_cell_num_atoms[4] == 1);
                REQUIRE(t_cell_num_atoms[5] == 1);
                REQUIRE(t_cell_num_atoms[6] == 3);
                REQUIRE(t_cell_num_atoms[7] == 0);
                REQUIRE(t_cell_num_atoms[8] == 3);
            }

            THEN("Then the global cell_cumulative_num_atoms = {0, 1, 1, 2, 2, 3, 4, 7, 7}") {
                REQUIRE(t_cell_cumulative_num_atoms[0] == 0);
                REQUIRE(t_cell_cumulative_num_atoms[1] == 1);
                REQUIRE(t_cell_cumulative_num_atoms[2] == 1);
                REQUIRE(t_cell_cumulative_num_atoms[3] == 2);
                REQUIRE(t_cell_cumulative_num_atoms[4] == 2);
                REQUIRE(t_cell_cumulative_num_atoms[5] == 3);
                REQUIRE(t_cell_cumulative_num_atoms[6] == 4);
                REQUIRE(t_cell_cumulative_num_atoms[7] == 7);
                REQUIRE(t_cell_cumulative_num_atoms[8] == 7);
            }
        }

        cudaFree(d_cell_id);
    }
}

SCENARIO("[DEVICE] Collide atoms", "[d-collide]") {
    GIVEN("An array of 10 atoms in a single cell.") {
        int num_atoms = 10;
        FN = 100;
        int num_cells = 1;
        double dt = 100*1.e-6;

        double3 vel[num_atoms];
        // Nothing special about these velcoities
        // They are just randomly generated for T=20uK
        vel[0] = make_double3( 0.034,-0.079, 0.006);
        vel[1] = make_double3(-0.012, 0.025,-0.012);
        vel[2] = make_double3(-0.044, 0.018,-0.031);
        vel[3] = make_double3( 0.064,-0.025,-0.009);
        vel[4] = make_double3(-0.006,-0.017, 0.017);
        vel[5] = make_double3(-0.000,-0.023, 0.052);
        vel[6] = make_double3( 0.063, 0.018, 0.069);
        vel[7] = make_double3( 0.021, 0.022, 0.002);
        vel[8] = make_double3(-0.032,-0.127,-0.074);
        vel[9] = make_double3( 0.066, 0.022, 0.075);
        double3 *d_vel;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_vel),
                                   num_atoms*sizeof(double3)));
        checkCudaErrors(cudaMemcpy(d_vel,
                                   vel,
                                   num_atoms*sizeof(double3),
                                   cudaMemcpyHostToDevice));

        int cell_id[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int *d_cell_id;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_id),
                                   num_atoms*sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_cell_id,
                                   cell_id,
                                   num_atoms*sizeof(int),
                                   cudaMemcpyHostToDevice));

        int cell_cumulative_num_atoms[2] = {0, 10};
        int *d_cell_cumulative_num_atoms;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_cumulative_num_atoms),
                                   (num_cells+1)*sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_cell_cumulative_num_atoms,
                                   cell_cumulative_num_atoms,
                                   (num_cells+1)*sizeof(int),
                                   cudaMemcpyHostToDevice));

        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_cells*sizeof(curandState)));
        initialise_rng_states(num_cells,
                              state);

        int *d_collision_count;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_collision_count),
                                   num_cells*sizeof(int)));
        checkCudaErrors(cudaMemset(d_collision_count,
                                   0,
                                   num_cells*sizeof(int)));

        double *d_collision_remainder;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_collision_remainder),
                                   total_num_cells*sizeof(double)));

        double sig_vr_max[1] = {sqrt(16.*kB*20.e-6/h_pi/mass)*cross_section};
        double *d_sig_vr_max;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sig_vr_max),
                                   num_cells*sizeof(double)));
        checkCudaErrors(cudaMemcpy(d_sig_vr_max,
                                   sig_vr_max,
                                   num_cells*sizeof(double),
                                   cudaMemcpyHostToDevice));

        // Make cell really small so that we can have collisions between the ten atoms
        copy_collision_params_to_device<<<1, 1>>>(make_double3(0., 0., 0.),
                                                  make_double3(2.5e-6, 2.5e-6, 2.5e-6),
                                                  make_int3(1,1,1),
                                                  FN);

        WHEN("The cu_collide function is called once") {

            cu_collide(num_cells,
                       d_cell_id,
                       d_cell_cumulative_num_atoms,
                       dt,
                       state,
                       d_collision_count,
                       d_collision_remainder,
                       d_sig_vr_max,
                       d_vel);

            int t_collision_count[num_cells];
            double t_sig_vr_max[num_cells];

            checkCudaErrors(cudaMemcpy(t_collision_count,
                                       d_collision_count,
                                       num_cells*sizeof(int),
                                       cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(t_sig_vr_max,
                                       d_sig_vr_max,
                                       num_cells*sizeof(double),
                                       cudaMemcpyDeviceToHost));

            THEN("We should expect zero simulated collisions") {
                REQUIRE(t_collision_count[0] == 0*FN);
            }
            THEN("The sig_vr_max array should not have been updated") {
                REQUIRE(t_sig_vr_max[0] == sig_vr_max[0]);
            }
        }

        cudaFree(d_vel);
        cudaFree(d_cell_id);
        cudaFree(d_cell_cumulative_num_atoms);
        cudaFree(state);
        cudaFree(d_collision_count);
        cudaFree(d_sig_vr_max);
    }
}

SCENARIO("[DEVICE] Collision rate", "[d-collrate]") {
    GIVEN("An array of 10,000 thermal atoms.") {
        int num_atoms = 1e5;
        FN = 10;
        
        // Initialise grid parameters
        k_num_cells = make_int3(35, 35, 35);
        total_num_cells = k_num_cells.x*k_num_cells.y*k_num_cells.z;
        
        double dt = 100*1.e-6;

        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_atoms*sizeof(curandState)));
        initialise_rng_states(num_atoms,
                              state);

        double3 *d_vel;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_vel),
                                   num_atoms*sizeof(double3)));
        // Generate velocity distribution
        generate_thermal_velocities(num_atoms,
                                    20.e-6,
                                    state,
                                    d_vel);

#if defined(IP)  // Ioffe Pritchard trap
        trap_geo trap_parameters;
        trap_parameters.B0 = 0.01;
        trap_parameters.dB = 20.;
        trap_parameters.ddB = 40000.;
#else  // Quadrupole trap
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;
#endif

        double3 *d_pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pos),
                                   num_atoms*sizeof(double3)));
        // Generate position distribution
        generate_thermal_positions(num_atoms,
                                   20.e-6,
                                   trap_parameters,
                                   state,
                                   d_pos);

        int *d_cell_id;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_id),
                                   num_atoms*sizeof(int)));
        checkCudaErrors(cudaMemset(d_cell_id,
                                   0,
                                   num_atoms*sizeof(int)));

        int *d_atom_id;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_atom_id),
                                   num_atoms*sizeof(int)));

        initialise_atom_id(num_atoms,
                           d_atom_id);

        int2 *d_cell_start_end;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_start_end),
                                   (total_num_cells+1)*sizeof(int2)));
        // checkCudaErrors(cudaMemset(d_cell_start_end,
        //                            -1,
        //                            (total_num_cells+1)*sizeof(int2)));
        negative_elements<<<total_num_cells+1,1>>>(total_num_cells+1,
                                                 d_cell_start_end);

        int *d_cell_num_atoms;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_num_atoms),
                                   (total_num_cells+1)*sizeof(int)));
        checkCudaErrors(cudaMemset(d_cell_num_atoms,
                                   0,
                                   (total_num_cells+1)*sizeof(int)));

        int *d_cell_cumulative_num_atoms;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cell_cumulative_num_atoms),
                                   (total_num_cells+1)*sizeof(int)));
        checkCudaErrors(cudaMemset(d_cell_cumulative_num_atoms,
                                   0,
                                   (total_num_cells+1)*sizeof(int)));

        int *d_collision_count;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_collision_count),
                                   total_num_cells*sizeof(int)));
        checkCudaErrors(cudaMemset(d_collision_count,
                                   0,
                                   total_num_cells*sizeof(int)));
        double *d_collision_remainder;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_collision_remainder),
                                   total_num_cells*sizeof(double)));
        // checkCudaErrors(cudaMemset(d_collision_remainder,
        //                            0.,
        //                            total_num_cells*sizeof(double)));
        zero_elements<<<total_num_cells,1>>>(total_num_cells,
                                             d_collision_remainder);

        double sig_vr_max[total_num_cells];
        for (int cell = 0; cell < total_num_cells; ++cell) {
             sig_vr_max[cell] = sqrt(16.*kB*20.e-6/h_pi/mass)*cross_section;
        }
        double *d_sig_vr_max;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sig_vr_max),
                                   total_num_cells*sizeof(double)));
        checkCudaErrors(cudaMemcpy(d_sig_vr_max,
                                   sig_vr_max,
                                   total_num_cells*sizeof(double),
                                   cudaMemcpyHostToDevice));

        cublasHandle_t cublas_handle;
        checkCudaErrors(cublasCreate(&cublas_handle));
        // Set up global grid parameters
        initialise_grid_params(num_atoms,
                               cublas_handle,
                               d_pos);

        WHEN("The collide_atoms function is called one hundred times") {
            for (int i = 0; i < 100; ++i) {
                collide_atoms(num_atoms,
                              total_num_cells,
                              dt,
                              d_pos,
                              d_vel,
                              state,
                              d_sig_vr_max,
                              d_cell_id,
                              d_atom_id,
                              d_cell_start_end,
                              d_cell_num_atoms,
                              d_cell_cumulative_num_atoms,
                              d_collision_remainder,
                              d_collision_count);
                progress_bar(i,
                             100);
            }

            int t_collision_count[total_num_cells];
            checkCudaErrors(cudaMemcpy(t_collision_count,
                                       d_collision_count,
                                       total_num_cells*sizeof(int),
                                       cudaMemcpyDeviceToHost));

            int total_coll = 0;
            for (int cell = 0; cell < total_num_cells; ++cell) {
                total_coll += t_collision_count[cell];
            }

#if defined(IP)  // Ioffe Pritchard trap
            THEN("We should expect the collision rate to agree with Walraven") {
                REQUIRE(total_coll < 2407 * (1+fractional_tol));
                REQUIRE(total_coll > 2407 * (1-fractional_tol));
            }
#else  // Quadrupole
            THEN("We should expect the collision rate to agree with Walraven") {
                REQUIRE(total_coll < 1026 * (1+fractional_tol));
                REQUIRE(total_coll > 1026 * (1-fractional_tol));
            }
#endif
        }

        checkCudaErrors(cublasDestroy(cublas_handle));

        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(state);
        cudaFree(d_sig_vr_max);
        cudaFree(d_cell_id);
        cudaFree(d_atom_id);
        cudaFree(d_cell_start_end);
        cudaFree(d_cell_num_atoms);
        cudaFree(d_cell_cumulative_num_atoms);
        cudaFree(d_collision_count);
        cudaFree(d_collision_remainder);
    }
}

__global__ void copy_d_grid_min(double3 *grid_min) {
    grid_min[0] = d_grid_min;
    return;
}

__global__ void copy_d_cell_length(double3 *cell_length) {
    cell_length[0] = d_cell_length;
    return;
}

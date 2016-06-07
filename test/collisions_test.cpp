/** \file
 *  \brief Unit tests for the collisions file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions_test.hpp"

double fractional_tol = 0.05; 

SCENARIO("[HOST] Initialise grid parameters", "[h-initgrid]") {
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

        k_num_cells = make_int3(2, 3, 4);

        WHEN("The initialise_grid_params function is called") {
            cublasHandle_t cublas_handle;
            initialise_grid_params(10,
                                   cublas_handle,
                                   pos);

            THEN("Then the global grid_min = {-10., -15., -10.} ") {
                REQUIRE(grid_min.x == -10.);
                REQUIRE(grid_min.y == -15.);
                REQUIRE(grid_min.z == -10.);
            }
            THEN("Then the global cell_length = {10., 10., 5.} ") {
                REQUIRE(cell_length.x == 10.);
                REQUIRE(cell_length.y == 10.);
                REQUIRE(cell_length.z == 5.);
            }
        }
    }
}

SCENARIO("[HOST] Index atoms", "[h-index]") {
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

        k_num_cells = make_int3(2, 3, 4);
        cublasHandle_t cublas_handle;
        initialise_grid_params(10,
                               cublas_handle,
                               pos);

        WHEN("The index_atoms function is called") {
            int test_cell_id[10] = {0};
            index_atoms(10,
                        pos,
                        test_cell_id);

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
        }
    }
}

SCENARIO("[HOST] Sort atoms", "[h-sort]") {
    GIVEN("An array of 10 known indices, with an associated atom_id array") {
        int atom_id[10] = {0,  1, 2,  3,  4,  5, 6,  7,  8, 9};
        int cell_id[10] = {15, 8, 15, 14, 24, 5, 24, 24, 3, 15};

        WHEN("The sort_atoms function is called") {
            sort_atoms(10,
                       cell_id,
                       atom_id);

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
    }
}

SCENARIO("[HOST] Count atoms", "[h-count]") {
    GIVEN("An array of 10 sorted cell_ids with num_cells = 8.") {
        int num_atoms = 10;
        int num_cells = 8;

        int cell_id[10] = {0, 2, 4, 5, 6, 6, 6, 8, 8, 8};

        WHEN("The sort_atoms function is called") {
            int t_cell_num_atoms[9] = {0};
            int t_cell_cumulative_num_atoms[9] = {0};

            int2 *t_cell_start_end;
            t_cell_start_end = reinterpret_cast<int2*>(calloc(num_cells+1,
                                                       sizeof(int2)));
            memset(t_cell_start_end,
                   -1,
                   (num_cells+1)*sizeof(int2));

            count_atoms(num_atoms,
                        num_cells,
                        cell_id,
                        t_cell_start_end,
                        t_cell_num_atoms,
                        t_cell_cumulative_num_atoms);

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

            free(t_cell_start_end);

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
    }
}

SCENARIO("[HOST] Collide atoms", "[h-collide]") {
    GIVEN("An array of 10 atoms in a single cell.") {
        int num_atoms = 10;
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

        int cell_id[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int atom_id[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int cell_cumulative_num_atoms[2] = {0, 10};

        pcg32_random_t *state;
        state = reinterpret_cast<pcg32_random_t*>(calloc(num_atoms,
                                                         sizeof(pcg32_random_t)));
        initialise_rng_states(num_cells,
                              state,
                              false);

        int *t_collision_count;
        t_collision_count = reinterpret_cast<int*>(calloc(num_cells,
                                                     sizeof(int)));

        double *collision_remainder;
        collision_remainder = reinterpret_cast<double*>(calloc(num_cells,
                                                     sizeof(double)));

        double sig_vr_max = sqrt(16.*kB*20.e-6/h_pi/mass)*cross_section;
        double t_sig_vr_max = sig_vr_max;

        // Make cell really small so that we can have collisions between the ten atoms
        grid_min = make_double3(0., 0., 0.);
        cell_length = make_double3(2.5e-6, 2.5e-6, 2.5e-6);
        cell_volume = cell_length.x * cell_length.y * cell_length.z;
        k_num_cells = make_int3(1, 1, 1);

        WHEN("The collide function is called once") {

            collide(num_cells,
                    cell_id,
                    atom_id,
                    cell_cumulative_num_atoms,
                    dt,
                    state,
                    t_collision_count,
                    collision_remainder,
                    &t_sig_vr_max,
                    vel);

            THEN("We should expect two simulated collisions") {
                REQUIRE(t_collision_count[0] == 0*FN);
            }
            THEN("The sig_vr_max array should not have been updated") {
                REQUIRE(t_sig_vr_max == sig_vr_max);
            }
        }

        free(state);
        free(t_collision_count);
    }
}

SCENARIO("[Host] Collision rate", "[h-collrate]") {
    GIVEN("An array of 1e5 thermal atoms.") {
        int num_atoms = 1e5;
        FN = 10;
        
        // Initialise grid parameters
        k_num_cells = make_int3(10, 10, 10);
        total_num_cells = k_num_cells.x*k_num_cells.y*k_num_cells.z;
        
        double dt = 100*1.e-6;

        pcg32_random_t *state;
        state = reinterpret_cast<pcg32_random_t*>(calloc(num_atoms,
                                                         sizeof(pcg32_random_t)));
        initialise_rng_states(num_atoms,
                              state,
                              false);

        double3 vel[num_atoms];
        // Generate velocity distribution
        generate_thermal_velocities(num_atoms,
                                    20.e-6,
                                    state,
                                    vel);

#if defined(IOFFE)  // Ioffe Pritchard trap
        trap_geo trap_parameters;
        trap_parameters.B0 = 0.01;
        trap_parameters.dB = 20.;
        trap_parameters.ddB = 40000.;
#else  // Quadrupole trap
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;
#endif

        double3 pos[num_atoms];
        // Generate position distribution
        generate_thermal_positions(num_atoms,
                                   20.e-6,
                                   trap_parameters,
                                   state,
                                   pos);

        int cell_id[num_atoms];
        memset(cell_id,
               0,
               num_atoms*sizeof(int));

        int atom_id[num_atoms];
        initialise_atom_id(num_atoms,
                           atom_id);

        int2 cell_start_end[total_num_cells];
        memset(cell_start_end,
               -1,
               (total_num_cells+1)*sizeof(int2));

        int cell_num_atoms[total_num_cells+1];
        memset(cell_num_atoms,
               0,
              (total_num_cells+1)*sizeof(int));

        int cell_cumulative_num_atoms[total_num_cells];
        memset(cell_cumulative_num_atoms,
               0,
               total_num_cells*sizeof(int));

        int t_collision_count[total_num_cells];
        memset(t_collision_count,
               0,
               total_num_cells*sizeof(int));

        double collision_remainder[total_num_cells];
        memset(collision_remainder,
               0.,
               total_num_cells*sizeof(double));

        double sig_vr_max[total_num_cells];
        for (int cell = 0; cell < total_num_cells; ++cell) {
             sig_vr_max[cell] = sqrt(16.*kB*20.e-6/h_pi/mass)*cross_section;
        }

        cublasHandle_t cublas_handle = NULL;
        // Set up global grid parameters
        initialise_grid_params(num_atoms,
                               cublas_handle,
                               pos);

        WHEN("The collide_atoms function is called one hundred times") {
            for (int i = 0; i < 100; ++i) {
                collide_atoms(num_atoms,
                              total_num_cells,
                              dt,
                              pos,
                              vel,
                              state,
                              sig_vr_max,
                              cell_id,
                              atom_id,
                              cell_start_end,
                              cell_num_atoms,
                              cell_cumulative_num_atoms,
                              collision_remainder,
                              t_collision_count);
                progress_bar(i,
                             100);
            }

            int total_coll = 0;
            for (int cell = 0; cell < total_num_cells; ++cell) {
                total_coll += t_collision_count[cell];
            }

#if defined(IOFFE)  // Ioffe Pritchard trap
            THEN("We should expect the collision rate to agree with Walraven") {
                REQUIRE(total_coll < 120 * (1+fractional_tol));
                REQUIRE(total_coll > 120 * (1-fractional_tol));
            }
#else  // Quadrupole
            THEN("We should expect the collision rate to agree with Walraven") {
                REQUIRE(total_coll < 1026 * (1+fractional_tol));
                REQUIRE(total_coll > 1026 * (1-fractional_tol));
            }
#endif
        }
    }
}

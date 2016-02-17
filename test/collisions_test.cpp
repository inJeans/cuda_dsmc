/** \file
 *  \brief Unit tests for the collisions file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions_test.hpp"

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

        num_cells = make_int3(2, 3, 4);

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

        num_cells = make_int3(2, 3, 4);
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

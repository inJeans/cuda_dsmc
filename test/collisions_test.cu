/** \file
 *  \brief Unit tests for the collisions file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions_test.cuh"

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

        num_cells = make_int3(2, 3, 4);

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

__global__ void copy_d_grid_min(double3 *grid_min) {
    grid_min[0] = d_grid_min;
    printf("{%f, %f, %f}\n", d_grid_min.x, d_grid_min.y, d_grid_min.z);
    printf("{%f, %f, %f}\n", grid_min[0].x,grid_min[0].y,grid_min[0].z);
    return;
}

__global__ void copy_d_cell_length(double3 *cell_length) {
    cell_length[0] = d_cell_length;
    return;
}
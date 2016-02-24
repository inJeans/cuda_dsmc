/** \file
 *  \brief file description
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <stdio.h>
#include <float.h>
#ifdef CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif

#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <iostream>
#include <iomanip>
#include <string>

#include "custom_sink.hpp"
#include "helper_cuda.h"
#include "define_host_constants.hpp"
#include "distribution_generation.hpp"
#include "distribution_evolution.hpp"
#include "collisions.hpp"

#define NUM_ATOMS 2

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    const std::string path_to_log_file = "./";
#else
    const std::string path_to_log_file = "/tmp/";
#endif

int main(int argc, char const *argv[]) {
    // Initialise logger
    auto worker = g3::LogWorker::createLogWorker();
    auto default_handle = worker->addDefaultLogger(argv[0], path_to_log_file);
    // auto output_handle = worker->addSink(std2::make_unique<CustomSink>(),
    //                                      &CustomSink::ReceiveLogMessage);
    g3::initializeLogging(worker.get());
    std::future<std::string> log_file_name = default_handle->
                                             call(&g3::FileSink::fileName);
    std::cout << "\n All logging output will be written to: "
              << log_file_name.get() << std::endl;
    g3::only_change_at_initialization::setLogLevel(DEBUG, false);

    printf("****************************\n");
    printf("*                          *\n");
    printf("*   WELCOME TO CUDA DSMC   *\n");
    printf("*                          *\n");
    printf("****************************\n");

#ifdef CUDA
    LOGF(INFO, "\nRunnning on your local CUDA device.");
    findCudaDevice(argc,
                   argv);
#endif

    // Initialise trapping parameters
    LOGF(INFO, "\nInitialising the trapping parameters.");
    trap_geo trap_parameters;
    trap_parameters.Bz = 2.0;
    trap_parameters.B0 = 0.;

    // Initialise computational parameters
    double dt = 1.e-6;
    int num_time_steps = 1;

    // Initialise grid parameters
    num_cells = make_int3(2, 2, 2);
    total_num_cells = num_cells.x*num_cells.y*num_cells.z;

    // Initialise rng
    LOGF(INFO, "\nInitialising the rng state array.");
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i curandState elements on the device.",
         NUM_ATOMS);
    curandState *state;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                               NUM_ATOMS*sizeof(curandState)));
#else
    LOGF(DEBUG, "\nAllocating %i pcg32_random_t elements on the host.",
         NUM_ATOMS);
    pcg32_random_t *state;
    state = reinterpret_cast<pcg32_random_t*>(calloc(NUM_ATOMS,
                                                     sizeof(pcg32_random_t)));
#endif
    initialise_rng_states(NUM_ATOMS,
                          state,
                          false);

    // Initialise atom_id
    LOGF(INFO, "\nInitialising the atom_id array.");
    int *atom_id;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         NUM_ATOMS);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&atom_id),
                               NUM_ATOMS*sizeof(int)));
#else
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         NUM_ATOMS);
    atom_id = reinterpret_cast<int*>(calloc(NUM_ATOMS,
                                            sizeof(int)));
#endif

    initialise_atom_id(NUM_ATOMS,
                       atom_id);

    // Initialise cell_id
    LOGF(INFO, "\nInitialising the cell_id array.");
    int *cell_id;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         NUM_ATOMS);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_id),
                               NUM_ATOMS*sizeof(int)));
    checkCudaErrors(cudaMemset(cell_id,
                               0,
                               NUM_ATOMS*sizeof(int)));
#else
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         NUM_ATOMS);
    cell_id = reinterpret_cast<int*>(calloc(NUM_ATOMS,
                                            sizeof(int)));
#endif

    // Initialise cell_start_end
    LOGF(INFO, "\nInitialising the cell_start_end array.");
    int2 *cell_start_end;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i int2 elements on the device.",
         total_num_cells+1);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_start_end),
                               (total_num_cells+1)*sizeof(int2)));
    checkCudaErrors(cudaMemset(cell_start_end,
                               -1,
                               (total_num_cells+1)*sizeof(int2)));
#else
    LOGF(DEBUG, "\nAllocating %i int2 elements on the host.",
         total_num_cells+1);
    cell_start_end = reinterpret_cast<int2*>(calloc(total_num_cells+1,
                                                    sizeof(int2)));
    memset(cell_start_end,
           -1,
           (total_num_cells+1)*sizeof(int2));
#endif

    // Initialise cell_num_atoms
    LOGF(INFO, "\nInitialising the cell_num_atoms array.");
    int *cell_num_atoms;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         total_num_cells+1);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_num_atoms),
                               (total_num_cells+1)*sizeof(int)));
    checkCudaErrors(cudaMemset(cell_num_atoms,
                               0,
                               (total_num_cells+1)*sizeof(int)));
#else
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         total_num_cells+1);
    cell_num_atoms = reinterpret_cast<int*>(calloc(total_num_cells+1,
                                                   sizeof(int)));
#endif

    // Initialise cell_cumulative_num_atoms
    LOGF(INFO, "\nInitialising the cell_cumulative_num_atoms array.");
    int *cell_cumulative_num_atoms;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         total_num_cells+1);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_cumulative_num_atoms),
                               (total_num_cells+1)*sizeof(int)));
    checkCudaErrors(cudaMemset(cell_num_atoms,
                               0,
                               (total_num_cells+1)*sizeof(int)));
#else
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         total_num_cells+1);
    cell_cumulative_num_atoms = reinterpret_cast<int*>(calloc(total_num_cells+1,
                                                       sizeof(int)));
#endif

    // Initialise velocities
    LOGF(INFO, "\nInitialising the velocity array.");
    double3 *vel;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
         NUM_ATOMS);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&vel),
                               NUM_ATOMS*sizeof(double3)));
#else
    LOGF(DEBUG, "\nAllocating %i double3 elements on the host.",
         NUM_ATOMS);
    vel = reinterpret_cast<double3*>(calloc(NUM_ATOMS,
                                            sizeof(double3)));
#endif

    // Generate velocity distribution
    generate_thermal_velocities(NUM_ATOMS,
                                20.e-6,
                                state,
                                vel);

    // Initialise positions
    LOGF(INFO, "\nInitialising the position array.");
    double3 *pos;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
         NUM_ATOMS);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&pos),
                               NUM_ATOMS*sizeof(double3)));
#else
    LOGF(DEBUG, "\nAllocating %i double3 elements on the host.",
         NUM_ATOMS);
    pos = reinterpret_cast<double3*>(calloc(NUM_ATOMS,
                                            sizeof(double3)));
#endif

    // Generate position distribution
    generate_thermal_positions(NUM_ATOMS,
                               20.e-6,
                               trap_parameters,
                               state,
                               pos);

    // Initialise accelerations
    LOGF(INFO, "\nInitialising the acceleration array.");
    double3 *acc;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
         NUM_ATOMS);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&acc),
                               NUM_ATOMS*sizeof(double3)));
#else
    LOGF(DEBUG, "\nAllocating %i double3 elements on the host.",
         NUM_ATOMS);
    acc = reinterpret_cast<double3*>(calloc(NUM_ATOMS,
                                            sizeof(double3)));
#endif

    // Generate accelerations
    update_accelerations(NUM_ATOMS,
                         trap_parameters,
                         pos,
                         acc);

// Initialise wavefunction
    LOGF(INFO, "\nInitialising the wavefunction array.");
    zomplex2 *psi;
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i zomplex2 elements on the device.",
         NUM_ATOMS);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&psi),
                               NUM_ATOMS*sizeof(zomplex2)));
#else
    LOGF(DEBUG, "\nAllocating %i zomplex2 elements on the host.",
         NUM_ATOMS);
    psi = reinterpret_cast<zomplex2*>(calloc(NUM_ATOMS,
                                             sizeof(zomplex2)));
#endif

    // Generate wavefunction
    generate_aligned_spins(NUM_ATOMS,
                           trap_parameters,
                           pos,
                           psi);

    LOGF(DEBUG, "\nBefore time evolution.\n");
#ifdef CUDA
    double3 h_vel[NUM_ATOMS];
    cudaMemcpy(&h_vel,
               vel,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);

    double3 h_pos[NUM_ATOMS];
    cudaMemcpy(&h_pos,
               pos,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);

    double3 h_acc[NUM_ATOMS];
    cudaMemcpy(&h_acc,
               acc,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);

    zomplex2 h_psi[NUM_ATOMS];
    cudaMemcpy(&h_psi,
               psi,
               NUM_ATOMS*sizeof(zomplex2),
               cudaMemcpyDeviceToHost);

    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", h_vel[0].x, h_vel[0].y, h_vel[0].z,
                                                           h_vel[1].x, h_vel[1].y, h_vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", h_pos[0].x, h_pos[0].y, h_pos[0].z,
                                                           h_pos[1].x, h_pos[1].y, h_pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", h_acc[0].x, h_acc[0].y, h_acc[0].z,
                                                           h_acc[1].x, h_acc[1].y, h_acc[1].z);
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi }, psi2 = { %f%+fi,%f%+fi }\n", 
               h_psi[0].up.x, h_psi[0].up.y, h_psi[0].dn.x, h_psi[0].dn.y,
               h_psi[1].up.x, h_psi[1].up.y, h_psi[1].dn.x, h_psi[1].dn.y);
#else 
    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
                                                           vel[1].x, vel[1].y, vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", pos[0].x, pos[0].y, pos[0].z,
                                                           pos[1].x, pos[1].y, pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", acc[0].x, acc[0].y, acc[0].z,
                                                           acc[1].x, acc[1].y, acc[1].z);
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi }, psi2 = { %f%+fi,%f%+fi }\n", 
               psi[0].up.x, psi[0].up.y, psi[0].dn.x, psi[0].dn.y,
               psi[1].up.x, psi[1].up.y, psi[1].dn.x, psi[1].dn.y);
#endif
    
    cublasHandle_t cublas_handle;
#ifdef CUDA
    LOGF(DEBUG, "\nCreating the cuBLAS handle.\n");
    checkCudaErrors(cublasCreate(&cublas_handle));
#endif

    // Set up global grid parameters
    initialise_grid_params(NUM_ATOMS,
                           cublas_handle,
                           pos);

    // Evolve many time step
    LOGF(INFO, "\nEvolving distribution for %i time steps.", num_time_steps);
    for (int i = 0; i < num_time_steps; ++i) {
        velocity_verlet_update(NUM_ATOMS,
                               dt,
                               trap_parameters,
                               cublas_handle,
                               pos,
                               vel,
                               acc);
        collide_atoms(NUM_ATOMS,
                      total_num_cells,
                      pos,
                      cell_id,
                      atom_id,
                      cell_start_end,
                      cell_num_atoms,
                      cell_cumulative_num_atoms);
    }
#ifdef CUDA
    LOGF(DEBUG, "\nDestroying the cuBLAS handle.\n");
    checkCudaErrors(cublasDestroy(cublas_handle));
#endif

    LOGF(DEBUG, "\nAfter time evolution.\n");
    #ifdef CUDA
    cudaMemcpy(&h_vel,
               vel,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pos,
               pos,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_acc,
               acc,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);

    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", h_vel[0].x, h_vel[0].y, h_vel[0].z,
                                                           h_vel[1].x, h_vel[1].y, h_vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", h_pos[0].x, h_pos[0].y, h_pos[0].z,
                                                           h_pos[1].x, h_pos[1].y, h_pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", h_acc[0].x, h_acc[0].y, h_acc[0].z,
                                                           h_acc[1].x, h_acc[1].y, h_acc[1].z);
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi }, psi2 = { %f%+fi,%f%+fi }\n", 
               h_psi[0].up.x, h_psi[0].up.y, h_psi[0].dn.x, h_psi[0].dn.y,
               h_psi[1].up.x, h_psi[1].up.y, h_psi[1].dn.x, h_psi[1].dn.y);
#else 
    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
                                                           vel[1].x, vel[1].y, vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", pos[0].x, pos[0].y, pos[0].z,
                                                           pos[1].x, pos[1].y, pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", acc[0].x, acc[0].y, acc[0].z,
                                                           acc[1].x, acc[1].y, acc[1].z);
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi }, psi2 = { %f%+fi,%f%+fi }\n", 
               psi[0].up.x, psi[0].up.y, psi[0].dn.x, psi[0].dn.y,
               psi[1].up.x, psi[1].up.y, psi[1].dn.x, psi[1].dn.y);
#endif

#ifdef CUDA
    LOGF(INFO, "\nCleaning up device memory.");
    cudaFree(state);
    cudaFree(atom_id);
    cudaFree(cell_id);
    cudaFree(cell_start_end);
    cudaFree(cell_num_atoms);
    cudaFree(cell_cumulative_num_atoms);
    cudaFree(vel);
    cudaFree(pos);
    cudaFree(acc);
    cudaFree(psi);
#else
    LOGF(INFO, "\nCleaning up local memory.");
    free(state);
    free(atom_id);
    free(cell_id);
    free(cell_start_end);
    free(cell_num_atoms);
    free(cell_cumulative_num_atoms);
    free(vel);
    free(pos);
    free(acc);
    free(psi);
#endif

    g3::internal::shutDownLogging();

    return 0;
}

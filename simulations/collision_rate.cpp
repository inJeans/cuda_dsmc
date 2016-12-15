/** \file
 *  \brief file description
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <stdio.h>
#include <float.h>
#if defined(CUDA)
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif

#if defined(LOGGING)
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#endif
#include <iostream>
#include <iomanip>
#include <string>

#if defined(LOGGING)
#include "custom_sink.hpp"
#endif
#include "helper_cuda.h"
#include "utilities.hpp"
#include "define_host_constants.hpp"
#include "distribution_generation.hpp"
#include "distribution_evolution.hpp"
#include "collisions.hpp"

#define NUM_ATOMS 100000

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    const std::string path_to_log_file = "./";
#else
    const std::string path_to_log_file = "/tmp/";
#endif

int main(int argc, char const *argv[]) {
#if defined(LOGGING)
    // Initialise logger
    auto worker = g3::LogWorker::createLogWorker();
    auto default_handle = worker->addDefaultLogger(argv[0], path_to_log_file);
    auto output_handle = worker->addSink(std2::make_unique<CustomSink>(),
                                         &CustomSink::ReceiveLogMessage);
    g3::initializeLogging(worker.get());
    std::future<std::string> log_file_name = default_handle->
                                             call(&g3::FileSink::fileName);
    std::cout << "\n All logging output will be written to: "
              << log_file_name.get() << std::endl;
    // g3::only_change_at_initialization::setLogLevel(DEBUG, false);
#endif
    printf("************************************\n");
    printf("*                                  *\n");
    printf("*   CUDA DSMC COLLISION RATE TEST  *\n");
    printf("*                                  *\n");
    printf("************************************\n");

#if defined(CUDA)
#if defined(LOGGING)
    LOGF(INFO, "\nRunnning on your local CUDA device.");
#endif
    findCudaDevice(argc,
                   argv);
#endif

    // Initialise trapping parameters
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the trapping parameters.");
#endif
#if defined(IOFFE)  // Ioffe Pritchard trap
    trap_geo trap_parameters;
    trap_parameters.B0 = 0.01;
    trap_parameters.dB = 20.;
    trap_parameters.ddB = 40000.;
#elif defined(QUAD)  // Quadrupole trap
    trap_geo trap_parameters;
    trap_parameters.Bz = 2.0;
    trap_parameters.B0 = 0.;
#else  // Harmonic trap
    trap_geo trap_parameters;
    trap_parameters.B0 = 0.;
    trap_parameters.wx = 450.;
    trap_parameters.wy = 450.;
    trap_parameters.wz = 450.;
#endif

    // Initialise computational parameters
    double dt = 1.e-6;
    int num_collision_steps = 10;
    int time_steps_per_collision = 1000;

    // Initialise grid parameters
    k_num_cells = make_int3(30, 30, 30);
    total_num_cells = k_num_cells.x*k_num_cells.y*k_num_cells.z;

    // Initialise rng
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the rng state array.");
#endif
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i curandState elements on the device.",
         NUM_ATOMS);
#endif
    curandState *state;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                               NUM_ATOMS*sizeof(curandState)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i pcg32_random_t elements on the host.",
         NUM_ATOMS);
#endif
    pcg32_random_t *state;
    state = reinterpret_cast<pcg32_random_t*>(calloc(NUM_ATOMS,
                                                     sizeof(pcg32_random_t)));
#endif
    initialise_rng_states(NUM_ATOMS,
                          state,
                          false);

    // Initialise atom_id
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the atom_id array.");
#endif
    int *atom_id;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         NUM_ATOMS);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&atom_id),
                               NUM_ATOMS*sizeof(int)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         NUM_ATOMS);
#endif
    atom_id = reinterpret_cast<int*>(calloc(NUM_ATOMS,
                                            sizeof(int)));
#endif

    initialise_atom_id(NUM_ATOMS,
                       atom_id);

    // Initialise cell_id
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the cell_id array.");
#endif
    int *cell_id;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         NUM_ATOMS);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_id),
                               NUM_ATOMS*sizeof(int)));
    checkCudaErrors(cudaMemset(cell_id,
                               0,
                               NUM_ATOMS*sizeof(int)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         NUM_ATOMS);
#endif
    cell_id = reinterpret_cast<int*>(calloc(NUM_ATOMS,
                                            sizeof(int)));
#endif

    // Initialise cell_start_end
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the cell_start_end array.");
#endif
    int2 *cell_start_end;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int2 elements on the device.",
         total_num_cells+1);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_start_end),
                               (total_num_cells+1)*sizeof(int2)));
    checkCudaErrors(cudaMemset(cell_start_end,
                               -1,
                               (total_num_cells+1)*sizeof(int2)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int2 elements on the host.",
         total_num_cells+1);
#endif
    cell_start_end = reinterpret_cast<int2*>(calloc(total_num_cells+1,
                                                    sizeof(int2)));
    memset(cell_start_end,
           -1,
           (total_num_cells+1)*sizeof(int2));
#endif

    // Initialise cell_num_atoms
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the cell_num_atoms array.");
#endif
    int *cell_num_atoms;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         total_num_cells+1);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_num_atoms),
                               (total_num_cells+1)*sizeof(int)));
    checkCudaErrors(cudaMemset(cell_num_atoms,
                               0,
                               (total_num_cells+1)*sizeof(int)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         total_num_cells+1);
#endif
    cell_num_atoms = reinterpret_cast<int*>(calloc(total_num_cells+1,
                                                   sizeof(int)));
#endif

    // Initialise cell_cumulative_num_atoms
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the cell_cumulative_num_atoms array.");
#endif
    int *cell_cumulative_num_atoms;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         total_num_cells+1);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_cumulative_num_atoms),
                               (total_num_cells+1)*sizeof(int)));
    checkCudaErrors(cudaMemset(cell_cumulative_num_atoms,
                               0,
                               (total_num_cells+1)*sizeof(int)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         total_num_cells+1);
#endif
    cell_cumulative_num_atoms = reinterpret_cast<int*>(calloc(total_num_cells+1,
                                                       sizeof(int)));
#endif

    // Initialise collision_count
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the collision_count array.");
#endif
    int *collision_count;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         total_num_cells);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&collision_count),
                               (total_num_cells)*sizeof(int)));
    checkCudaErrors(cudaMemset(collision_count,
                               0,
                               (total_num_cells)*sizeof(int)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         total_num_cells);
#endif
    collision_count = reinterpret_cast<int*>(calloc(total_num_cells,
                                                       sizeof(int)));
#endif

    // Initialise collision_remainder
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the collision_remainder array.");
#endif
    double *collision_remainder;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         total_num_cells);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&collision_remainder),
                               (total_num_cells)*sizeof(double)));
    checkCudaErrors(cudaMemset(collision_remainder,
                               0.,
                               (total_num_cells)*sizeof(double)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         total_num_cells);
#endif
    collision_remainder = reinterpret_cast<double*>(calloc(total_num_cells,
                                                        sizeof(double)));
#endif

    // Initialise sig_vr_max
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the sig_vr_max array.");
#endif
    double *sig_vr_max;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the device.",
         total_num_cells);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&sig_vr_max),
                               (total_num_cells)*sizeof(double)));
    checkCudaErrors(cudaMemset(sig_vr_max,
                               0.,
                               (total_num_cells)*sizeof(double)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i int elements on the host.",
         total_num_cells);
#endif
    sig_vr_max = reinterpret_cast<double*>(calloc(total_num_cells,
                                               sizeof(double)));
#endif

    // Initialise velocities
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the velocity array.");
#endif
    double3 *vel;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
         NUM_ATOMS);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&vel),
                               NUM_ATOMS*sizeof(double3)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i double3 elements on the host.",
         NUM_ATOMS);
#endif
    vel = reinterpret_cast<double3*>(calloc(NUM_ATOMS,
                                            sizeof(double3)));
#endif

    // Generate velocity distribution
    generate_thermal_velocities(NUM_ATOMS,
                                20.e-6,
                                state,
                                vel);

    // Initialise positions
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the position array.");
#endif
    double3 *pos;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
         NUM_ATOMS);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&pos),
                               NUM_ATOMS*sizeof(double3)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i double3 elements on the host.",
         NUM_ATOMS);
#endif
    pos = reinterpret_cast<double3*>(calloc(NUM_ATOMS,
                                            sizeof(double3)));
#endif

    // Generate position distribution
    generate_thermal_positions(NUM_ATOMS,
                               20.e-6,
                               trap_parameters,
                               state,
                               pos);

#if defined(SPIN)
    // Initialise wavefunction
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the wavefunction array.");
#endif
    wavefunction *psi;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i wavefunction elements on the device.",
         NUM_ATOMS);
#endif  // Logging
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&psi),
                               NUM_ATOMS*sizeof(wavefunction)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i wavefunction elements on the host.",
         NUM_ATOMS);
#endif  // Logging
    psi = reinterpret_cast<wavefunction*>(calloc(NUM_ATOMS,
                                             sizeof(wavefunction)));
#endif // CUDA

    // Generate wavefunction
    generate_aligned_spins(NUM_ATOMS,
                           trap_parameters,
                           pos,
                           psi);
#endif // Spin


    // Initialise accelerations
#if defined(LOGGING)
    LOGF(INFO, "\nInitialising the acceleration array.");
#endif
    double3 *acc;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
         NUM_ATOMS);
#endif
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&acc),
                               NUM_ATOMS*sizeof(double3)));
#else
#if defined(LOGGING)
    LOGF(DEBUG, "\nAllocating %i double3 elements on the host.",
         NUM_ATOMS);
#endif
    acc = reinterpret_cast<double3*>(calloc(NUM_ATOMS,
                                            sizeof(double3)));
#endif

#if defined(SPIN)
    // Generate accelerations
    update_accelerations(NUM_ATOMS,
                         trap_parameters,
                         pos,
                         acc,
                         psi);
#else
    // Generate accelerations
    update_accelerations(NUM_ATOMS,
                         trap_parameters,
                         pos,
                         acc);
#endif // Spin

    cublasHandle_t cublas_handle;
#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nCreating the cuBLAS handle.\n");
#endif
    checkCudaErrors(cublasCreate(&cublas_handle));
#endif

    // Set up global grid parameters
    initialise_grid_params(NUM_ATOMS,
                           cublas_handle,
                           pos);

#if defined(CUDA)
    int *h_collision_count;
    checkCudaErrors(cudaHostAlloc(reinterpret_cast<void **>(&h_collision_count),
                                  total_num_cells*sizeof(int),
                                  cudaHostAllocDefault));
    memset(h_collision_count,
           0,
           total_num_cells*sizeof(int));

    FILE *collision_file_pointer = fopen("collision.data", "w");
    for (int cell=0; cell < total_num_cells; ++cell)
        fprintf(collision_file_pointer, "%i\t", h_collision_count[cell]);

    fprintf(collision_file_pointer, "\n");
    fclose(collision_file_pointer);
#endif

    // Evolve many time step
#if defined(LOGGING)
    LOGF(INFO, "\nEvolving distribution for %i time steps.", num_collision_steps * time_steps_per_collision);
#endif
    for (int i = 0; i < num_collision_steps; ++i) {
#if defined(SPIN)
        velocity_verlet_evolution(NUM_ATOMS,
                                  dt,
                                  time_steps_per_collision,
                                  trap_parameters,
                                  cublas_handle,
                                  pos,
                                  vel,
                                  acc,
                                  psi);
#else
        velocity_verlet_evolution(NUM_ATOMS,
                                  dt,
                                  time_steps_per_collision,
                                  trap_parameters,
                                  cublas_handle,
                                  pos,
                                  vel,
                                  acc);
#endif // Spin
        collide_atoms(NUM_ATOMS,
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
                     collision_count);

#if defined(CUDA)
        checkCudaErrors(cudaMemcpy(h_collision_count,
                                   collision_count,
                                   total_num_cells*sizeof(int),
                                   cudaMemcpyDeviceToHost));
        collision_file_pointer = fopen("collision.data", "a");
        for (int cell=0; cell < total_num_cells; ++cell)
            fprintf(collision_file_pointer, "%i\t", h_collision_count[cell]);
        fprintf(collision_file_pointer, "\n");
        fclose(collision_file_pointer);
#endif
        progress_bar(i,
                     num_collision_steps-1);
    }
#if defined(CUDA)
        checkCudaErrors(cudaMemcpy(h_collision_count,
                                   collision_count,
                                   total_num_cells*sizeof(int),
                                   cudaMemcpyDeviceToHost));
        int total_num_collisions = 0;
        for (int cell=0; cell < total_num_cells; ++cell)
            total_num_collisions += h_collision_count[cell];
        double total_time = time_steps_per_collision * num_collision_steps * dt;
        double collision_rate = (double) total_num_collisions / NUM_ATOMS / total_time;
        printf("Collision rate estimate = %f /s\n", collision_rate );
#endif

#if defined(CUDA)
#if defined(LOGGING)
    LOGF(DEBUG, "\nDestroying the cuBLAS handle.\n");
#endif
    checkCudaErrors(cublasDestroy(cublas_handle));
#endif

#if defined(CUDA)
#if defined(LOGGING)
    LOGF(INFO, "\nCleaning up device memory.");
#endif
    cudaFree(state);
    cudaFree(atom_id);
    cudaFree(cell_id);
    cudaFree(cell_start_end);
    cudaFree(cell_num_atoms);
    cudaFree(cell_cumulative_num_atoms);
    cudaFree(collision_count);
    cudaFree(collision_remainder);
    cudaFree(sig_vr_max);
    cudaFree(vel);
    cudaFree(pos);
    cudaFree(acc);

    cudaFreeHost(h_collision_count);
#if defined(SPIN)
    cudaFree(psi);
#endif // Spin
#else
#if defined(LOGGING)
    LOGF(INFO, "\nCleaning up local memory.");
#endif
    free(state);
    free(atom_id);
    free(cell_id);
    free(cell_start_end);
    free(cell_num_atoms);
    free(cell_cumulative_num_atoms);
    free(collision_count);
    free(collision_remainder);
    free(sig_vr_max);
    free(vel);
    free(pos);
    free(acc);
#if defined(SPIN)
    free(psi);
#endif  // Spin
#endif  // CUDA
#if defined(LOGGING)
    g3::internal::shutDownLogging();
#endif

    return 0;
}

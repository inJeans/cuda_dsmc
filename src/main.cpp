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
    auto output_handle = worker->addSink(std2::make_unique<CustomSink>(),
                                         &CustomSink::ReceiveLogMessage);
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

#if defined(CUDA)
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
    double dt = 1.e-3;
    int num_time_steps = 2;

    // Initialise rng
    LOGF(INFO, "\nInitialising the rng state array.");
#if defined(CUDA)
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

    // Initialise velocities
    LOGF(INFO, "\nInitialising the velocity array.");
    double3 *vel;
#if defined(CUDA)
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
#if defined(CUDA)
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

#if defined(SPIN)
    // Initialise wavefunction
    LOGF(INFO, "\nInitialising the wavefunction array.");
    wavefunction *psi;
#if defined(CUDA)
    LOGF(DEBUG, "\nAllocating %i wavefunction elements on the device.",
         NUM_ATOMS);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&psi),
                               NUM_ATOMS*sizeof(wavefunction)));
#else
    LOGF(DEBUG, "\nAllocating %i wavefunction elements on the host.",
         NUM_ATOMS);
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
    LOGF(INFO, "\nInitialising the acceleration array.");
    double3 *acc;
#if defined(CUDA)
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

    LOGF(DEBUG, "\nBefore time evolution.\n");
#if defined(CUDA)
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

#if defined(SPIN)
    wavefunction h_psi[NUM_ATOMS];
    cudaMemcpy(&h_psi,
               psi,
               NUM_ATOMS*sizeof(wavefunction),
               cudaMemcpyDeviceToHost);
#endif // Spin

    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", h_vel[0].x, h_vel[0].y, h_vel[0].z,
                                                           h_vel[1].x, h_vel[1].y, h_vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", h_pos[0].x, h_pos[0].y, h_pos[0].z,
                                                           h_pos[1].x, h_pos[1].y, h_pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", h_acc[0].x, h_acc[0].y, h_acc[0].z,
                                                           h_acc[1].x, h_acc[1].y, h_acc[1].z);
#if defined(SPIN)
#if defined(EHRENFEST)
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi, }, psi2 = { %f%+fi,%f%+fi }\n", 
               h_psi[0].up.x, h_psi[0].up.y, h_psi[0].dn.x, h_psi[0].dn.y,
               h_psi[1].up.x, h_psi[1].up.y, h_psi[1].dn.x, h_psi[1].dn.y);
#else
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi %d }, psi2 = { %f%+fi,%f%+fi %d }\n", 
               h_psi[0].up.x, h_psi[0].up.y, h_psi[0].dn.x, h_psi[0].dn.y, h_psi[0].isSpinUp,
               h_psi[1].up.x, h_psi[1].up.y, h_psi[1].dn.x, h_psi[1].dn.y, h_psi[1].isSpinUp);
#endif // Ehrenfest
#endif // Spin
#else 
    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
                                                           vel[1].x, vel[1].y, vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", pos[0].x, pos[0].y, pos[0].z,
                                                           pos[1].x, pos[1].y, pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", acc[0].x, acc[0].y, acc[0].z,
                                                           acc[1].x, acc[1].y, acc[1].z);
#if defined(SPIN)
#if defined(EHRENFEST)
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi, }, psi2 = { %f%+fi,%f%+fi }\n", 
               psi[0].up.x, psi[0].up.y, psi[0].dn.x, psi[0].dn.y,
               psi[1].up.x, psi[1].up.y, psi[1].dn.x, psi[1].dn.y);
#else
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi %d }, psi2 = { %f%+fi,%f%+fi %d }\n", 
               psi[0].up.x, psi[0].up.y, psi[0].dn.x, psi[0].dn.y, psi[0].isSpinUp,
               psi[1].up.x, psi[1].up.y, psi[1].dn.x, psi[1].dn.y, psi[1].isSpinUp);
#endif // Ehrenfest
#endif // Spin
#endif

    cublasHandle_t cublas_handle;
#if defined(CUDA)
    LOGF(DEBUG, "\nCreating the cuBLAS handle.\n");
    checkCudaErrors(cublasCreate(&cublas_handle));
#endif
    // Evolve many time step
    LOGF(INFO, "\nEvolving distribution for %i time steps.", num_time_steps);
    for (int i = 0; i < num_time_steps; ++i) {
#if defined(SPIN)
        velocity_verlet_update(NUM_ATOMS,
                               dt,
                               trap_parameters,
                               cublas_handle,
                               pos,
                               vel,
                               acc,
                               psi);
#else
        velocity_verlet_update(NUM_ATOMS,
                               dt,
                               trap_parameters,
                               cublas_handle,
                               pos,
                               vel,
                               acc);
#endif // Spin
    }
#if defined(CUDA)
    LOGF(DEBUG, "\nDestroying the cuBLAS handle.\n");
    cublasDestroy(cublas_handle);
#endif

    LOGF(DEBUG, "\nAfter time evolution.\n");
#if defined(CUDA)
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
#if defined(SPIN)
    cudaMemcpy(&h_psi,
               psi,
               NUM_ATOMS*sizeof(wavefunction),
               cudaMemcpyDeviceToHost);
#endif // Spin

    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", h_vel[0].x, h_vel[0].y, h_vel[0].z,
                                                           h_vel[1].x, h_vel[1].y, h_vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", h_pos[0].x, h_pos[0].y, h_pos[0].z,
                                                           h_pos[1].x, h_pos[1].y, h_pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", h_acc[0].x, h_acc[0].y, h_acc[0].z,
                                                           h_acc[1].x, h_acc[1].y, h_acc[1].z);
#if defined(SPIN)
#if defined(EHRENFEST)
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi, }, psi2 = { %f%+fi,%f%+fi }\n", 
               h_psi[0].up.x, h_psi[0].up.y, h_psi[0].dn.x, h_psi[0].dn.y,
               h_psi[1].up.x, h_psi[1].up.y, h_psi[1].dn.x, h_psi[1].dn.y);
#else
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi %d }, psi2 = { %f%+fi,%f%+fi %d }\n", 
               h_psi[0].up.x, h_psi[0].up.y, h_psi[0].dn.x, h_psi[0].dn.y, h_psi[0].isSpinUp,
               h_psi[1].up.x, h_psi[1].up.y, h_psi[1].dn.x, h_psi[1].dn.y, h_psi[1].isSpinUp);
#endif // Ehrenfest
#endif // Spin
#else 
    LOGF(INFO, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
                                                           vel[1].x, vel[1].y, vel[1].z);
    LOGF(INFO, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", pos[0].x, pos[0].y, pos[0].z,
                                                           pos[1].x, pos[1].y, pos[1].z);
    LOGF(INFO, "\na1 = { %f,%f,%f }, a2 = { %f,%f,%f }\n", acc[0].x, acc[0].y, acc[0].z,
                                                           acc[1].x, acc[1].y, acc[1].z);
#if defined(SPIN)
#if defined(EHRENFEST)
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi, }, psi2 = { %f%+fi,%f%+fi }\n", 
               psi[0].up.x, psi[0].up.y, psi[0].dn.x, psi[0].dn.y,
               psi[1].up.x, psi[1].up.y, psi[1].dn.x, psi[1].dn.y);
#else
    LOGF(INFO, "\npsi1 = { %f%+fi,%f%+fi %d }, psi2 = { %f%+fi,%f%+fi %d }\n", 
               psi[0].up.x, psi[0].up.y, psi[0].dn.x, psi[0].dn.y, psi[0].isSpinUp,
               psi[1].up.x, psi[1].up.y, psi[1].dn.x, psi[1].dn.y, psi[1].isSpinUp);
#endif // Ehrenfest
#endif // Spin
#endif

#if defined(CUDA)
    LOGF(INFO, "\nCleaning up device memory.");
    cudaFree(state);
    cudaFree(vel);
    cudaFree(pos);
    cudaFree(acc);
#if defined(SPIN)
    cudaFree(psi);
#endif // Spin
#else
    LOGF(INFO, "\nCleaning up local memory.");
    free(state);
    free(vel);
    free(pos);
    free(acc);
#if defined(SPIN)
    free(psi);
#endif // Spin
#endif // CUDA

    g3::internal::shutDownLogging();

    return 0;
}

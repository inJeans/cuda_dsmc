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
#endif

#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <iostream>
#include <iomanip>
#include <string>

#include "custom_sink.hpp"
#include "helper_cuda.h"
#include "distribution_generation.hpp"

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
    // g3::only_change_at_initialization::setLogLevel(INFO, false);

    printf("****************************\n");
    printf("*                          *\n");
    printf("*   WELCOME TO CUDA DSMC   *\n");
    printf("*                          *\n");
    printf("****************************\n");

#ifdef CUDA
    LOGF(INFO, "\nRunnning on your local CUDA device.");
#endif

    // Initialise trapping parameters
    LOGF(INFO, "\nInitialising the trapping parameters.");
    trap_geo trap_parameters;
    trap_parameters.Bz = 2.0;
    trap_parameters.B0 = 0.;

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

    // Generate velocity distribution
    generate_thermal_positions(NUM_ATOMS,
                               20.e-6,
                               trap_parameters,
                               state,
                               pos);

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

    LOGF(DEBUG, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", h_vel[0].x, h_vel[0].y, h_vel[0].z,
                                                            h_vel[1].x, h_vel[1].y, h_vel[1].z);
    LOGF(DEBUG, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", h_pos[0].x, h_pos[0].y, h_pos[0].z,
                                                            h_pos[1].x, h_pos[1].y, h_pos[1].z);
    cudaFree(h_vel);
    cudaFree(h_pos);
#else 
    LOGF(DEBUG, "\nv1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
                                                            vel[1].x, vel[1].y, vel[1].z);
    LOGF(DEBUG, "\np1 = { %f,%f,%f }, p2 = { %f,%f,%f }\n", pos[0].x, pos[0].y, pos[0].z,
                                                            pos[1].x, pos[1].y, pos[1].z);
#endif

#ifdef CUDA
    LOGF(INFO, "\nCleaning up device memory.");
    cudaFree(state);
    cudaFree(vel);
    cudaFree(pos);
#else
    LOGF(INFO, "\nCleaning up local memory.");
    free(state);
    free(vel);
    free(pos);
#endif

    g3::internal::shutDownLogging();

    return 0;
}

/** \fn void doxygen_test( double x )
 *  \brief Short description
 *  \param x double that gets printed
 *  \warning What does this do?
 *  Detailed description starts here.
 *  \return void
 */

void doxygen_test(double x) {
    printf("%f\n", x);
    return;
}

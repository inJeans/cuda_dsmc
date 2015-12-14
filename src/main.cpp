/** \file
 *  \brief file description
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

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
    printf("****************************\n");
    printf("*                          *\n");
    printf("*   WELCOME TO CUDA DSMC   *\n");
    printf("*                          *\n");
    printf("****************************\n");

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

#ifdef CUDA
    LOGF(INFO, "\nRunnning on your local CUDA device.");
#endif

    // Initialise rng
    LOGF(INFO, "\nInitialising the rng state array.");
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i curandState elements on the device.",
         NUM_ATOMS);
    curandState *state;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                               NUM_ATOMS*sizeof(curandState)));
#else
    LOGF(DEBUG, "\nAllocating %i pcg64_random_t elements on the host.",
         NUM_ATOMS);
    pcg64_random_t *state;
    state = reinterpret_cast<pcg64_random_t*>(calloc(NUM_ATOMS,
                                                     sizeof(pcg64_random_t)));
#endif
    initialise_rng_states(NUM_ATOMS,
                          state);

    // Initialise velocities
    LOGF(INFO, "\nInitialising the velocity array.");
#ifdef CUDA
    LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
         NUM_ATOMS);
    double3 *vel;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&vel),
                               NUM_ATOMS*sizeof(double3)));
#else
    LOGF(DEBUG, "\nAllocating %i double3 elements on the host.",
         NUM_ATOMS);
    double3 *vel;
    vel = reinterpret_cast<double3*>(calloc(NUM_ATOMS,
                                            sizeof(double3)));
#endif

    // Generate distribution
    generate_thermal_velocities(NUM_ATOMS,
                                20.e-6,
                                state,
                                vel);

#ifdef CUDA
    double3 h_vel[NUM_ATOMS];
    cudaMemcpy(&h_vel,
               vel,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);

    printf("v1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", h_vel[0].x, h_vel[0].y, h_vel[0].z,
                                                     h_vel[1].x, h_vel[1].y, h_vel[1].z);
#else 
    printf("v1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
                                                     vel[1].x, vel[1].y, vel[1].z);
#endif

    // cudaFree(state);
    // cudaFree(d_vel);

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

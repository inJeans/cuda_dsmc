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

#include "custom_sink.hpp"
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
    auto handle = worker->addDefaultLogger(argv[0], path_to_log_file);
    auto handle_term = worker->addSink(std2::make_unique<CustomSink>(),
                                       &CustomSink::ReceiveLogMessage);
    g3::initializeLogging(worker.get());
    std::future<std::string> log_file_name = handle->call(&g3::FileSink::fileName);
    std::cout << "\n All logging output will be written to: "
              << log_file_name.get() << std::endl;
    // g3::only_change_at_initialization::setLogLevel(INFO, false);

    // Initialise rng
    LOGF(INFO, "\nInitialising the rng state arrays.\n");
#ifdef CUDA
    curandState *state;
    cudaMalloc(reinterpret_cast<void **>(&state),
               NUM_ATOMS*sizeof(curandState));
#else
    pcg64_random_t *state;
    state = reinterpret_cast<pcg64_random_t*>(calloc(NUM_ATOMS,
                                                     sizeof(pcg64_random_t)));
#endif
    initialise_rng_states(NUM_ATOMS,
                          state);

    // double3 *d_vel;
    // cudaMalloc(reinterpret_cast<void **>(&d_vel),
    //            NUM_ATOMS*sizeof(double3));


    // // Generate distribution
    // generate_thermal_velocities(NUM_ATOMS,
    //                             20.e-6,
    //                             state,
    //                             vel);

    // double3 vel[NUM_ATOMS];
    // cudaMemcpy(&vel,
    //            vel,
    //            NUM_ATOMS*sizeof(double3),
    //            cudaMemcpyDeviceToHost);

    // printf("v1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
    //                                                  vel[1].x, vel[1].y, vel[1].z);

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

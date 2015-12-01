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

#include "distribution_generation.hpp"

#define NUM_ATOMS 2

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    const std::string path_to_log_file = "./";
#else
    const std::string path_to_log_file = "/tmp/";
#endif

struct CustomSink {

// Linux xterm color
// http://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal
  enum FG_Color {YELLOW = 33, RED = 31, GREEN=32, WHITE=97, CYAN=36};

  FG_Color GetColor(const LEVELS level) const {
     if (level.value == WARNING.value) { return YELLOW; }
     if (level.value == DEBUG.value) { return GREEN; }
     if (g3::internal::wasFatal(level)) { return RED; }

     return CYAN;
  }

  void ReceiveLogMessage(g3::LogMessageMover logEntry) {
     auto level = logEntry.get()._level;
     auto color = GetColor(level);

     std::cout << "\033[" << color << "m"
               << logEntry.get().toString() << "\033[m" << std::endl;
  }
};

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
    curandState *d_state;
    cudaMalloc(reinterpret_cast<void **>(&d_state),
               NUM_ATOMS*sizeof(curandState));
    initialise_rng_states(NUM_ATOMS,
                          d_state);

    double3 *d_vel;
    cudaMalloc(reinterpret_cast<void **>(&d_vel),
               NUM_ATOMS*sizeof(double3));

    // Generate distribution
    generate_thermal_velocities(NUM_ATOMS,
                                20.e-6,
                                d_state,
                                d_vel);

    double3 vel[NUM_ATOMS];
    cudaMemcpy(&vel,
               d_vel,
               NUM_ATOMS*sizeof(double3),
               cudaMemcpyDeviceToHost);

    printf("v1 = { %f,%f,%f }, v2 = { %f,%f,%f }\n", vel[0].x, vel[0].y, vel[0].z,
                                                     vel[1].x, vel[1].y, vel[1].z);

    cudaFree(d_state);
    cudaFree(d_vel);

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

/** \file
 *  \brief Main code
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#include <stdio.h>
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <g3log/std2_make_unique.hpp>

#include <string>
#include <math.h>

#include "gtest/gtest.h"

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/define_physical_constants.hpp"
 #include "cuda_dsmc/declare_physical_constants.cuh"
#include "cuda_dsmc/define_physical_constants.cuh"
#include "cuda_dsmc/dsmc_utils.cuh"
#include "cuda_dsmc/vector_math.cuh"

#include "cuda_dsmc/magnetic_field.cuh"
#include "cuda_dsmc/distribution_generation.cuh"
#include "cuda_dsmc/particle_evolution.cuh"
#include "cuda_dsmc/CustomSink.hpp"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    mkdir("./tmp", 0700);
    const std::string kPathToLogFile = "./tmp/";
#else
    const std::string kPathToLogFile = "/tmp/";
#endif
const std::string kLogfilename = "test_dsmc_distribution_generation";

int kRNGSeed = 1234;
int kNumAtoms = 1e3;
int kNumTestLoops = 1e6;
double kDt = 1.e-6;
double kTestTemp = 100.e-9;

double kTolerance = 5. / sqrt(kNumAtoms);

double kExpectedKineticEnergy = 0.5 * kKB * kTestTemp;
double kKineticEnergyStdDev = 0.5 * sqrt(21.) * kKB * kTestTemp;

int kNumBlocks = 1024;
int kNumThreads = 128;

#if defined(HARMONIC)
FieldParams kTestParams = {.omega = make_double3(1., 1., 1.),
                           .B0 = 0.
                          };
double kExpectedPotentialEnergy = 1.;
double3 kTestPosStdDev = make_double3(1., 1., 1.);
#else  // No magnetic field
FieldParams kTestParams = {.B0 = 0.,
                           .max_distribution_width = 1.};
double kExpectedPotentialEnergy = 0.;
double3 kTestPosStdDev = make_double3(0.539, 0.539, 0.539);
#endif

class EvolutionTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        /* Get device count */
        CUDA_CALL(cudaGetDeviceCount(&num_devices));
        LOGF(INFO, "... %i device(s) detected", num_devices);

        d_streams = reinterpret_cast<cudaStream_t *>(calloc(num_devices, sizeof(cudaStream_t)));
        d_states = reinterpret_cast<curandState **>(calloc(num_devices, sizeof(curandState*)));

        for (int d = 0; d < num_devices; ++d) {
            CUDA_CALL(cudaSetDevice(d));
            CUDA_CALL(cudaStreamCreate(&d_streams[d]));
            /* Allocate space for rng states on device */
            CUDA_CALL(cudaMalloc((void **)&d_states[d],
                                 kNumBlocks * kNumThreads * sizeof(curandState)));
        }

        /* Initialise rng states on device */
        initRNG(kNumBlocks*kNumThreads,
                kRNGSeed,
                d_streams,
                d_states);

        /* Allocate kNumAtoms double3s on host */
        h_pos = reinterpret_cast<double3 *>(calloc(kNumAtoms, sizeof(double3)));
        h_vel = reinterpret_cast<double3 *>(calloc(kNumAtoms, sizeof(double3)));

        // Initialise distributions
        generateThermalPositionDistribution(kNumAtoms,
                                            kTestParams,
                                            kTestTemp,
                                            d_streams,
                                            d_states,
                                            &d_pos);
        generateThermalVelocityDistribution(kNumAtoms,
                                            kTestTemp,
                                            d_streams,
                                            d_states,
                                            &d_vel);

        /* Copy device memory to host */ 
        combineDeviceArrays(num_devices,
                            kNumAtoms,
                            d_pos,
                            h_pos);
        combineDeviceArrays(num_devices,
                            kNumAtoms,
                            d_vel,
                            h_vel);
    }

    virtual void TearDown() {
        for (int d=0; d < num_devices; ++d) {
            cudaFree(d_vel[d]);
        }
        free(d_vel);
        free(d_streams);
        free(d_states);
        free(h_vel);
    }

    cudaStream_t *d_streams;
    curandState **d_states;

    int num_devices;

    double3 *h_pos, *h_vel;
    double3 **d_pos, **d_vel;
};

TEST_F(EvolutionTest, KineticEnergyMean) {
    double3 initial_directional_mean;
    double initial_global_mean = kineticEnergyMean(h_vel,
                                                   kNumAtoms,
                                                   &initial_directional_mean);

    ASSERT_LT(initial_directional_mean.x, (1.+kTolerance) * kExpectedKineticEnergy);
    ASSERT_LT(initial_directional_mean.y, (1.+kTolerance) * kExpectedKineticEnergy);
    ASSERT_LT(initial_directional_mean.z, (1.+kTolerance) * kExpectedKineticEnergy);
    ASSERT_LT(initial_global_mean, (1.+kTolerance) * kExpectedKineticEnergy * 3.);

    ASSERT_GT(initial_directional_mean.x, (1.-kTolerance) * kExpectedKineticEnergy);
    ASSERT_GT(initial_directional_mean.y, (1.-kTolerance) * kExpectedKineticEnergy);
    ASSERT_GT(initial_directional_mean.z, (1.-kTolerance) * kExpectedKineticEnergy);
    ASSERT_GT(initial_global_mean, (1.-kTolerance) * kExpectedKineticEnergy * 3.);

    for (int loop=0; loop < kNumTestLoops; loop++)
        evolveParticleDistribution(kNumAtoms,
                                   kTestParams,
                                   kDt,
                                   d_streams,
                                   &d_pos,
                                   &d_vel);


    /* Copy device memory to host */ 
    combineDeviceArrays(num_devices,
                        kNumAtoms,
                        d_vel,
                        h_vel);

    double3 final_directional_mean;
    double final_global_mean = kineticEnergyMean(h_vel,
                                                 kNumAtoms,
                                                 &final_directional_mean);

    ASSERT_LT(final_directional_mean.x, (1.+kTolerance) * kExpectedKineticEnergy);
    ASSERT_LT(final_directional_mean.y, (1.+kTolerance) * kExpectedKineticEnergy);
    ASSERT_LT(final_directional_mean.z, (1.+kTolerance) * kExpectedKineticEnergy);
    ASSERT_LT(final_global_mean, (1.+kTolerance) * kExpectedKineticEnergy * 3.);

    ASSERT_GT(final_directional_mean.x, (1.-kTolerance) * kExpectedKineticEnergy);
    ASSERT_GT(final_directional_mean.y, (1.-kTolerance) * kExpectedKineticEnergy);
    ASSERT_GT(final_directional_mean.z, (1.-kTolerance) * kExpectedKineticEnergy);
    ASSERT_GT(final_global_mean, (1.-kTolerance) * kExpectedKineticEnergy * 3.);

    double3 directional_error = abs(final_directional_mean - initial_directional_mean) / 
                                    initial_directional_mean;
    double global_error = abs(final_global_mean - initial_global_mean) / 
                              initial_global_mean;

    ASSERT_LT(directional_error.x, kTolerance);
    ASSERT_LT(directional_error.y, kTolerance);
    ASSERT_LT(directional_error.z, kTolerance);
    ASSERT_LT(global_error, kTolerance);

    ASSERT_GT(directional_error.x, -1. * kTolerance);
    ASSERT_GT(directional_error.y, -1. * kTolerance);
    ASSERT_GT(directional_error.z, -1. * kTolerance);
    ASSERT_GT(global_error, -1. * kTolerance);
}

TEST_F(EvolutionTest, KineticEnergyStdDev) {
    double3 initial_directional_stddev;
    double initial_global_stddev = kineticEnergyStddev(h_vel,
                                                       kNumAtoms,
                                                       &initial_directional_stddev);

    // ASSERT_LT(initial_directional_stddev.x, kKineticEnergyStdDev * (1. + kTolerance));
    // ASSERT_LT(initial_directional_stddev.y, kKineticEnergyStdDev * (1. + kTolerance));
    // ASSERT_LT(initial_directional_stddev.z, kKineticEnergyStdDev * (1. + kTolerance));
    ASSERT_LT(initial_global_stddev, kKineticEnergyStdDev * (1. + kTolerance));

    // ASSERT_GT(initial_directional_stddev.x, kKineticEnergyStdDev * (1. - kTolerance));
    // ASSERT_GT(initial_directional_stddev.y, kKineticEnergyStdDev * (1. - kTolerance));
    // ASSERT_GT(initial_directional_stddev.z, kKineticEnergyStdDev * (1. - kTolerance));
    ASSERT_GT(initial_global_stddev, kKineticEnergyStdDev * (1. - kTolerance));

    for (int loop=0; loop < kNumTestLoops; loop++)
        evolveParticleDistribution(kNumAtoms,
                                   kTestParams,
                                   kDt,
                                   d_streams,
                                   &d_pos,
                                   &d_vel);

    /* Copy device memory to host */ 
    combineDeviceArrays(num_devices,
                        kNumAtoms,
                        d_vel,
                        h_vel);

    double3 final_directional_stddev;
    double final_global_stddev = kineticEnergyStddev(h_vel,
                                                     kNumAtoms,
                                                     &final_directional_stddev);

    // ASSERT_LT(final_directional_stddev.x, (1.+kTolerance) * kExpectedKineticEnergy);
    // ASSERT_LT(final_directional_stddev.y, (1.+kTolerance) * kExpectedKineticEnergy);
    // ASSERT_LT(final_directional_stddev.z, (1.+kTolerance) * kExpectedKineticEnergy);
    ASSERT_LT(final_global_stddev, (1.+kTolerance) * kKineticEnergyStdDev);

    // ASSERT_GT(final_directional_stddev.x, (1.-kTolerance) * kExpectedKineticEnergy);
    // ASSERT_GT(final_directional_stddev.y, (1.-kTolerance) * kExpectedKineticEnergy);
    // ASSERT_GT(final_directional_stddev.z, (1.-kTolerance) * kExpectedKineticEnergy);
    ASSERT_GT(final_global_stddev, (1.-kTolerance) * kKineticEnergyStdDev);

    double3 directional_error = abs(final_directional_stddev - initial_directional_stddev) / 
                                    initial_directional_stddev;
    double global_error = abs(final_global_stddev - initial_global_stddev) / 
                              initial_global_stddev;

    ASSERT_LT(directional_error.x, kTolerance);
    ASSERT_LT(directional_error.y, kTolerance);
    ASSERT_LT(directional_error.z, kTolerance);
    ASSERT_LT(global_error, kTolerance);

    ASSERT_GT(directional_error.x, -1. * kTolerance);
    ASSERT_GT(directional_error.y, -1. * kTolerance);
    ASSERT_GT(directional_error.z, -1. * kTolerance);
    ASSERT_GT(global_error, -1. * kTolerance);
}

TEST_F(EvolutionTest, PotentialEnergyMean) {
    double3 initial_directional_mean;
    double initial_global_mean = potentialEnergyMean(h_pos,
                                                     kNumAtoms,
                                                     kTestParams,
                                                     &initial_directional_mean);

    ASSERT_LE(initial_directional_mean.x, (1.+kTolerance) * kExpectedPotentialEnergy);
    ASSERT_LE(initial_directional_mean.y, (1.+kTolerance) * kExpectedPotentialEnergy);
    ASSERT_LE(initial_directional_mean.z, (1.+kTolerance) * kExpectedPotentialEnergy);
    ASSERT_LE(initial_global_mean, (1.+kTolerance) * kExpectedPotentialEnergy * 3.);

    ASSERT_GE(initial_directional_mean.x, (1.-kTolerance) * kExpectedPotentialEnergy);
    ASSERT_GE(initial_directional_mean.y, (1.-kTolerance) * kExpectedPotentialEnergy);
    ASSERT_GE(initial_directional_mean.z, (1.-kTolerance) * kExpectedPotentialEnergy);
    ASSERT_GE(initial_global_mean, (1.-kTolerance) * kExpectedPotentialEnergy * 3.);

    for (int loop=0; loop < kNumTestLoops; loop++)
        evolveParticleDistribution(kNumAtoms,
                                   kTestParams,
                                   kDt,
                                   d_streams,
                                   &d_pos,
                                   &d_vel);

    /* Copy device memory to host */ 
    combineDeviceArrays(num_devices,
                        kNumAtoms,
                        d_pos,
                        h_pos);


    double3 final_directional_mean;
    double final_global_mean = potentialEnergyMean(h_pos,
                                                   kNumAtoms,
                                                   kTestParams,
                                                   &final_directional_mean);

    ASSERT_LE(final_directional_mean.x, (1.+kTolerance) * kExpectedPotentialEnergy);
    ASSERT_LE(final_directional_mean.y, (1.+kTolerance) * kExpectedPotentialEnergy);
    ASSERT_LE(final_directional_mean.z, (1.+kTolerance) * kExpectedPotentialEnergy);
    ASSERT_LE(final_global_mean, (1.+kTolerance) * kExpectedPotentialEnergy * 3.);

    ASSERT_GE(final_directional_mean.x, (1.-kTolerance) * kExpectedPotentialEnergy);
    ASSERT_GE(final_directional_mean.y, (1.-kTolerance) * kExpectedPotentialEnergy);
    ASSERT_GE(final_directional_mean.z, (1.-kTolerance) * kExpectedPotentialEnergy);
    ASSERT_GE(final_global_mean, (1.-kTolerance) * kExpectedPotentialEnergy * 3.);

    double3 directional_error = make_double3(0., 0., 0.);
    if (initial_directional_mean.x != 0 && 
        initial_directional_mean.y != 0 &&
        initial_directional_mean.z != 0) {
        directional_error = abs(final_directional_mean - initial_directional_mean) / 
                                initial_directional_mean;
    }
    else if (final_directional_mean.x != 0 && 
             final_directional_mean.y != 0 &&
             final_directional_mean.z != 0) {
        directional_error = abs(final_directional_mean - initial_directional_mean) / 
                                final_directional_mean;
    }
    else {
        directional_error = abs(final_directional_mean - initial_directional_mean);
    }

    double global_error = 0.;
    if (initial_global_mean != 0) {
        global_error = abs(final_global_mean - initial_global_mean) / 
                           initial_global_mean;
    }
    else if (final_global_mean != 0) {
        global_error = abs(final_global_mean - initial_global_mean) / 
                           final_global_mean;
    }
    else {
        global_error = abs(final_global_mean - initial_global_mean);
    }

    ASSERT_LT(directional_error.x, kTolerance);
    ASSERT_LT(directional_error.y, kTolerance);
    ASSERT_LT(directional_error.z, kTolerance);
    ASSERT_LT(global_error, kTolerance);

    ASSERT_GT(directional_error.x, -1. * kTolerance);
    ASSERT_GT(directional_error.y, -1. * kTolerance);
    ASSERT_GT(directional_error.z, -1. * kTolerance);
    ASSERT_GT(global_error, -1. * kTolerance);
}

TEST_F(DistributionTest, PositionStdDev) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += h_pos[test].x;
        test_sum.y += h_pos[test].y;
        test_sum.z += h_pos[test].z;
    }
#if defined(DSMC_MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &test_sum,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    double3 test_mean = test_sum / kNumAtoms;
    double global_mean = (test_mean.x + test_mean.y + test_mean.x) / 3.;

    // double3 sum_of_squared_differences = make_double3(0., 0., 0.);
    // for (int test = 0; test < kNumAtoms; ++test) {
    //     sum_of_squared_differences.x += (pos[test].x - test_mean.x) *
    //                                     (pos[test].x - test_mean.x);
    //     sum_of_squared_differences.y += (pos[test].y - test_mean.y) *
    //                                     (pos[test].y - test_mean.y);
    //     sum_of_squared_differences.z += (pos[test].z - test_mean.z) *
    //                                     (pos[test].z - test_mean.z);
    // }
// #if defined(DSMC_MPI)
//     MPI_Allreduce(MPI_IN_PLACE,
//                   &sum_of_squared_differences,
//                   3,
//                   MPI_DOUBLE,
//                   MPI_SUM,
//                   MPI_COMM_WORLD);
// #endif
//     double3 test_std_dev = make_double3(0., 0., 0.);
//     test_std_dev.x = sqrt(sum_of_squared_differences.x / (kNumAtoms-1));
//     test_std_dev.y = sqrt(sum_of_squared_differences.y / (kNumAtoms-1));
//     test_std_dev.z = sqrt(sum_of_squared_differences.z / (kNumAtoms-1));
//     double global_std_dev = (test_std_dev.x +
//                              test_std_dev.y +
//                              test_std_dev.z) / 3.;

//     ASSERT_LT(test_std_dev.x, kTestPosStdDev.x * (1. + kTolerance));
//     ASSERT_LT(test_std_dev.y, kTestPosStdDev.y * (1. + kTolerance));
//     ASSERT_LT(test_std_dev.z, kTestPosStdDev.z * (1. + kTolerance));
//     // ASSERT_LT(global_std_dev, kTestPosStdDev * (1. + kTolerance));

//     ASSERT_GT(test_std_dev.x, kTestPosStdDev.x * (1. - kTolerance));
//     ASSERT_GT(test_std_dev.y, kTestPosStdDev.y * (1. - kTolerance));
//     ASSERT_GT(test_std_dev.z, kTestPosStdDev.z * (1. - kTolerance));
//     // ASSERT_GT(global_std_dev, kTestPosStdDev * (1. - kTolerance));
}

int main(int argc, char **argv) {
    auto worker = g3::LogWorker::createLogWorker();
    auto logfileHandle = worker->addDefaultLogger(kLogfilename,
                                                  kPathToLogFile);

    // logger is initialized
    g3::initializeLogging(worker.get());

    auto stdoutHandle = worker->addSink(std2::make_unique<CustomSink>(),
                                        &CustomSink::ReceiveLogMessage);

    LOGF(INFO, "Testing distribution generators.");
#if defined(DSMC_MPI)
    // Initialize the MPI environment
    printf("Initialising MPI\n");
    MPI_Init(&argc, &argv);
#endif

    ::testing::InitGoogleTest(&argc, argv);

    int return_value = RUN_ALL_TESTS();

#if defined(DSMC_MPI)
    // Finalize the MPI environment.
    MPI_Finalize();
#endif
    return return_value;
}

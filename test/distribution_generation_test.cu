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
#include <curand.h>
#include <curand_kernel.h>

#include "gtest/gtest.h"

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/declare_physical_constants.cuh"
#include "cuda_dsmc/define_physical_constants.cuh"
#include "cuda_dsmc/define_physical_constants.hpp"
#include "cuda_dsmc/vector_math.cuh"

#include "cuda_dsmc/distribution_generation.cuh"
#include "cuda_dsmc/CustomSink.hpp"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    mkdir("./tmp", 0700);
    const std::string kPathToLogFile = "./tmp/";
#else
    const std::string kPathToLogFile = "/tmp/";
#endif
const std::string kLogfilename = "test_cuda_dsmc_distribution_generation";

int kRNGSeed = 1234;
double kNumAtoms = 1.e4;
double kTestTemp = 100.e-9;
double kTestV = sqrt(kKB * kTestTemp / kMass);

double kTolerance = 5. / sqrt(kNumAtoms);

int kNumBlocks = 1024;
int kNumThreads = 128;

#if defined(HARMONIC)
FieldParams kTestParams = {.omega = make_double3(1., 1., 1.),
                           .B0 = 0.
                          };
double3 kTestPosStdDev = make_double3(1., 1., 1.);
double kTestPosGlobalStdDev = sqrt(3.);
#else  // No magnetic field
FieldParams kTestParams = {.B0 = 0.};
double3 kTestPosStdDev = make_double3(0.539, 0.539, 0.539);
double kTestPosGlobalStdDev = sqrt(3.)*0.539;
#endif

class DeviceDistributionTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
#if defined(MPI)
        // Initialize the MPI environment
        printf("Initialising MPI\n");
        MPI_Init(NULL, NULL);
#endif
        /* Allocate space for rng states on device */
        CUDA_CALL(cudaMalloc((void **)&d_states,
                             kNumBlocks * kNumThreads * sizeof(curandState)));
        /* Initialise rng states on device */
        initRNG(kNumBlocks*kNumThreads,
                kRNGSeed,
                d_states);

        /* Allocate kNumAtoms double3s on host */
        h_pos = reinterpret_cast<double3 *>(calloc(kNumAtoms, sizeof(double3)));
        h_vel = reinterpret_cast<double3 *>(calloc(kNumAtoms, sizeof(double3)));

        // Initialise distributions
        generateThermalPositionDistribution(kNumAtoms,
                                            kTestParams,
                                            kTestTemp,
                                            d_states,
                                            &d_pos);
        generateThermalVelocityDistribution(kNumAtoms,
                                            kTestTemp,
                                            d_states,
                                            &d_vel);

        /* Copy device memory to host */ 
        CUDA_CALL(cudaMemcpy(h_pos, d_pos, kNumAtoms * sizeof(double3), cudaMemcpyDeviceToHost)); 
        CUDA_CALL(cudaMemcpy(h_vel, d_vel, kNumAtoms * sizeof(double3), cudaMemcpyDeviceToHost)); 
    }

    virtual void TearDown() {
        cudaFree(d_states);
        cudaFree(d_pos);
        cudaFree(d_vel);
        free(h_pos);
        free(h_vel);

#if defined(MPI)
        // Finalize the MPI environment.
        MPI_Finalize();
#endif
    }

    curandState *d_states;

    double3 *h_pos, *h_vel;
    double3 *d_pos, *d_vel;
};

TEST_F(DeviceDistributionTest, VelocityMean) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += h_vel[test].x;
        test_sum.y += h_vel[test].y;
        test_sum.z += h_vel[test].z;
    }
    double3 test_mean = test_sum / kNumAtoms;
    double global_mean = (test_mean.x + test_mean.y + test_mean.x) / 3.;

    ASSERT_LT(test_mean.x, kTolerance);
    ASSERT_LT(test_mean.y, kTolerance);
    ASSERT_LT(test_mean.z, kTolerance);
    ASSERT_LT(global_mean, kTolerance);

    ASSERT_GT(test_mean.x, -1. * kTolerance);
    ASSERT_GT(test_mean.y, -1. * kTolerance);
    ASSERT_GT(test_mean.z, -1. * kTolerance);
    ASSERT_GT(global_mean, -1. * kTolerance);
}

TEST_F(DeviceDistributionTest, VelocityStdDev) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += h_vel[test].x;
        test_sum.y += h_vel[test].y;
        test_sum.z += h_vel[test].z;
    }
    double3 test_mean = test_sum / kNumAtoms;

    double3 sum_of_squared_differences = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        sum_of_squared_differences.x += (h_vel[test].x - test_mean.x) *
                                        (h_vel[test].x - test_mean.x);
        sum_of_squared_differences.y += (h_vel[test].y - test_mean.y) *
                                        (h_vel[test].y - test_mean.y);
        sum_of_squared_differences.z += (h_vel[test].z - test_mean.z) *
                                        (h_vel[test].z - test_mean.z);
    }
    double3 test_std_dev = make_double3(0., 0., 0.);
    test_std_dev.x = sqrt(sum_of_squared_differences.x / (kNumAtoms-1));
    test_std_dev.y = sqrt(sum_of_squared_differences.y / (kNumAtoms-1));
    test_std_dev.z = sqrt(sum_of_squared_differences.z / (kNumAtoms-1));
    double global_std_dev = (test_std_dev.x +
                             test_std_dev.y +
                             test_std_dev.z) / 3.;

    ASSERT_LT(test_std_dev.x, kTestV * (1. + kTolerance));
    ASSERT_LT(test_std_dev.y, kTestV * (1. + kTolerance));
    ASSERT_LT(test_std_dev.z, kTestV * (1. + kTolerance));
    ASSERT_LT(global_std_dev, kTestV * (1. + kTolerance));

    ASSERT_GT(test_std_dev.x, kTestV * (1. - kTolerance));
    ASSERT_GT(test_std_dev.y, kTestV * (1. - kTolerance));
    ASSERT_GT(test_std_dev.z, kTestV * (1. - kTolerance));
    ASSERT_GT(global_std_dev, kTestV * (1. - kTolerance));
}

TEST_F(DeviceDistributionTest, PositionMean) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += h_pos[test].x;
        test_sum.y += h_pos[test].y;
        test_sum.z += h_pos[test].z;
    }
    double3 test_mean = test_sum / kNumAtoms;
    double global_mean = (test_mean.x + test_mean.y + test_mean.x) / 3.;

    ASSERT_LT(test_mean.x, kTolerance);
    ASSERT_LT(test_mean.y, kTolerance);
    ASSERT_LT(test_mean.z, kTolerance);
    ASSERT_LT(global_mean, kTolerance);

    ASSERT_GT(test_mean.x, -1. * kTolerance);
    ASSERT_GT(test_mean.y, -1. * kTolerance);
    ASSERT_GT(test_mean.z, -1. * kTolerance);
    ASSERT_GT(global_mean, -1. * kTolerance);
}

TEST_F(DeviceDistributionTest, PositionStdDev) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += h_pos[test].x;
        test_sum.y += h_pos[test].y;
        test_sum.z += h_pos[test].z;
    }
    double3 test_mean = test_sum / kNumAtoms;

    double3 sum_of_squared_differences = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        sum_of_squared_differences.x += (h_pos[test].x - test_mean.x) *
                                        (h_pos[test].x - test_mean.x);
        sum_of_squared_differences.y += (h_pos[test].y - test_mean.y) *
                                        (h_pos[test].y - test_mean.y);
        sum_of_squared_differences.z += (h_pos[test].z - test_mean.z) *
                                        (h_pos[test].z - test_mean.z);
    }
    double3 test_std_dev = make_double3(0., 0., 0.);
    test_std_dev.x = sqrt(sum_of_squared_differences.x / (kNumAtoms-1));
    test_std_dev.y = sqrt(sum_of_squared_differences.y / (kNumAtoms-1));
    test_std_dev.z = sqrt(sum_of_squared_differences.z / (kNumAtoms-1));
    double global_std_dev = sqrt(test_std_dev.x*test_std_dev.x +
                                 test_std_dev.y*test_std_dev.y +
                                 test_std_dev.z*test_std_dev.z);

    ASSERT_LT(test_std_dev.x, kTestPosStdDev.x * (1. + kTolerance));
    ASSERT_LT(test_std_dev.y, kTestPosStdDev.y * (1. + kTolerance));
    ASSERT_LT(test_std_dev.z, kTestPosStdDev.z * (1. + kTolerance));
    ASSERT_LT(global_std_dev, kTestPosGlobalStdDev * (1. + kTolerance));

    ASSERT_GT(test_std_dev.x, kTestPosStdDev.x * (1. - kTolerance));
    ASSERT_GT(test_std_dev.y, kTestPosStdDev.y * (1. - kTolerance));
    ASSERT_GT(test_std_dev.z, kTestPosStdDev.z * (1. - kTolerance));
    ASSERT_GT(global_std_dev, kTestPosGlobalStdDev * (1. - kTolerance));
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

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

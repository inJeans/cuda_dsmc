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
int kNumAtoms = 1e4;
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
FieldParams kTestParams = {.B0 = 0.,
                           .max_distribution_width = 1.};
double3 kTestPosStdDev = make_double3(0.539, 0.539, 0.539);
double kTestPosGlobalStdDev = sqrt(3.)*0.539;
#endif

class DeviceVelocityDistributionTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
#if defined(MPI)
        // Initialize the MPI environment
        printf("Initialising MPI\n");
        MPI_Init(NULL, NULL);
#endif
        /* Get device count */
        CUDA_CALL(cudaGetDeviceCount(&num_devices));

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
        h_vel = reinterpret_cast<double3 *>(calloc(kNumAtoms, sizeof(double3)));

        // Initialise distributions
        generateThermalVelocityDistribution(kNumAtoms,
                                            kTestTemp,
                                            d_streams,
                                            d_states,
                                            &d_vel);

        /* Copy device memory to host */ 
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

#if defined(MPI)
        // Finalize the MPI environment.
        MPI_Finalize();
#endif
    }

    cudaStream_t *d_streams;
    curandState **d_states;

    int num_devices;

    double3 *h_vel;
    double3 **d_vel;
};

TEST_F(DeviceVelocityDistributionTest, VelocityMean) {
    double3 directional_mean;
    double global_mean = mean(h_vel,
                              kNumAtoms,
                              &directional_mean);

    ASSERT_LT(directional_mean.x, kTolerance);
    ASSERT_LT(directional_mean.y, kTolerance);
    ASSERT_LT(directional_mean.z, kTolerance);
    ASSERT_LT(global_mean, kTolerance);

    ASSERT_GT(directional_mean.x, -1. * kTolerance);
    ASSERT_GT(directional_mean.y, -1. * kTolerance);
    ASSERT_GT(directional_mean.z, -1. * kTolerance);
    ASSERT_GT(global_mean, -1. * kTolerance);
}

TEST_F(DeviceVelocityDistributionTest, VelocityStdDev) {
    double3 directional_stddev;
    double global_stddev = stddev(h_vel,
                                  kNumAtoms,
                                  &directional_stddev);

    ASSERT_LT(directional_stddev.x, kTestV * (1. + kTolerance));
    ASSERT_LT(directional_stddev.y, kTestV * (1. + kTolerance));
    ASSERT_LT(directional_stddev.z, kTestV * (1. + kTolerance));
    ASSERT_LT(global_stddev, kTestV * (1. + kTolerance));

    ASSERT_GT(directional_stddev.x, kTestV * (1. - kTolerance));
    ASSERT_GT(directional_stddev.y, kTestV * (1. - kTolerance));
    ASSERT_GT(directional_stddev.z, kTestV * (1. - kTolerance));
    ASSERT_GT(global_stddev, kTestV * (1. - kTolerance));
}

class DevicePositionDistributionTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
#if defined(MPI)
        // Initialize the MPI environment
        printf("Initialising MPI\n");
        MPI_Init(NULL, NULL);
#endif
        /* Get device count */
        CUDA_CALL(cudaGetDeviceCount(&num_devices));

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

        // Initialise distributions
        generateThermalPositionDistribution(kNumAtoms,
                                            kTestParams,
                                            kTestTemp,
                                            d_streams,
                                            d_states,
                                            &d_pos);

        /* Copy device memory to host */ 
        combineDeviceArrays(num_devices,
                            kNumAtoms,
                            d_pos,
                            h_pos); 
    }

    virtual void TearDown() {
        for (int d = 0; d < num_devices; ++d) {
            cudaFree(d_pos[d]);
        }
        free(d_pos);
        free(d_streams);
        free(d_states);
        free(h_pos);

#if defined(MPI)
        // Finalize the MPI environment.
        MPI_Finalize();
#endif
    }

    cudaStream_t *d_streams;
    curandState **d_states;

    int num_devices;

    double3 *h_pos;
    double3 **d_pos;
};

TEST_F(DevicePositionDistributionTest, PositionMean) {
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

TEST_F(DevicePositionDistributionTest, PositionStdDev) {
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

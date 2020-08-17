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
int kNumAtoms = 1e7;
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
double3 kTestPosStdDev = make_double3(0.577, 0.577, 0.577);
double kTestPosGlobalStdDev = sqrt(3.)*0.577;
#endif

#if defined(DSMC_MPI)
ncclComm_t kcomm;
#endif

class DeviceVelocityDistributionTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        // CUDA_CHECK(cudaStreamCreate(&s));

        CUDA_CHECK(cudaMalloc((void **)&d_states,
                              kNumBlocks * kNumThreads * sizeof(curandState)));

        /* Initialise rng states on device */
        initRNG(kNumBlocks*kNumThreads,
                kRNGSeed,
                d_states);

        /* Allocate kNumAtoms double3s on host */
        h_vel = reinterpret_cast<double3 *>(calloc(kNumAtoms, sizeof(double3)));

        // Initialise distributions
        generateThermalVelocityDistribution(kNumAtoms,
                                            kTestTemp,
                                            d_states,
                                            &d_vel);

        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(h_vel, d_vel, 
                              kNumAtoms * sizeof(double3),
                              cudaMemcpyDeviceToHost)); 
    }

    virtual void TearDown() {
        cudaFree(d_states);
        cudaFree(d_vel);
        
        free(h_vel);
    }

    cudaStream_t s;

    curandState *d_states;

    double3 *h_vel;
    double3 *d_vel;
};

TEST_F(DeviceVelocityDistributionTest, VelocityMean) {
    double3 directional_mean;
    double global_mean = mean(h_vel,
                              kNumAtoms,
                              &directional_mean);

    ASSERT_LT(directional_mean.x/2, kTolerance);
    ASSERT_LT(directional_mean.y/2, kTolerance);
    ASSERT_LT(directional_mean.z/2, kTolerance);
    ASSERT_LT(global_mean, kTolerance);

    ASSERT_GT(directional_mean.x/2, -1. * kTolerance);
    ASSERT_GT(directional_mean.y/2, -1. * kTolerance);
    ASSERT_GT(directional_mean.z/2, -1. * kTolerance);
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

        CUDA_CHECK(cudaMalloc((void **)&d_states,
                              kNumBlocks * kNumThreads * sizeof(curandState)));

        /* Initialise rng states on device */
        initRNG(kNumBlocks*kNumThreads,
                kRNGSeed,
                d_states);

        /* Allocate kNumAtoms double3s on host */
        h_pos = reinterpret_cast<double3 *>(calloc(kNumAtoms, sizeof(double3)));

        // Initialise distributions
        generateThermalPositionDistribution(kNumAtoms,
                                            kTestParams,
                                            kTestTemp,
                                            d_states,
                                            &d_pos);

        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(h_pos, d_pos, 
                              kNumAtoms * sizeof(double3),
                              cudaMemcpyDeviceToHost)); 
    }

    virtual void TearDown() {
        cudaFree(d_pos);

        free(h_pos);
    }

    curandState *d_states;

    double3 *h_pos;
    double3 *d_pos;
};

TEST_F(DevicePositionDistributionTest, PositionMean) {
    double3 directional_mean;
    double global_mean = mean(h_pos,
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

TEST_F(DevicePositionDistributionTest, PositionStdDev) {
    double3 directional_stddev;
    double global_stddev = stddev(h_pos,
                                  kNumAtoms,
                                  &directional_stddev);

    ASSERT_LT(directional_stddev.x, kTestPosStdDev.x * (1. + kTolerance));
    ASSERT_LT(directional_stddev.y, kTestPosStdDev.y * (1. + kTolerance));
    ASSERT_LT(directional_stddev.z, kTestPosStdDev.z * (1. + kTolerance));
    ASSERT_LT(global_stddev, kTestPosGlobalStdDev * (1. + kTolerance));

    ASSERT_GT(directional_stddev.x, kTestPosStdDev.x * (1. - kTolerance));
    ASSERT_GT(directional_stddev.y, kTestPosStdDev.y * (1. - kTolerance));
    ASSERT_GT(directional_stddev.z, kTestPosStdDev.z * (1. - kTolerance));
    ASSERT_GT(global_stddev, kTestPosGlobalStdDev * (1. - kTolerance));
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
    int world_size, world_rank;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    LOGF(INFO, "... seting device id\n");
    int device_id = getLocalDeviceId();
    LOGF(INFO, "... device id set %i\n", device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    ncclUniqueId id;

    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (world_rank == 0) ncclGetUniqueId(&id);
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //initializing NCCL
    NCCL_CHECK(ncclCommInitRank(&kcomm, world_size, id, world_rank));
#endif

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

#if defined(DSMC_MPI)
    //finalizing NCCL
    NCCL_CHECK(ncclCommDestroy(kcomm));

    // Finalize the MPI environment.
    MPI_CHECK(MPI_Finalize());
#endif

    return result;
}

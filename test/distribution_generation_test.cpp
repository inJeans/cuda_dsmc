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

#include "gtest/gtest.h"

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/define_physical_constants.hpp"
#include "cuda_dsmc/vector_math.cuh"
#include "cuda_dsmc/dsmc_utils.hpp"

#include "cuda_dsmc/magnetic_field.hpp"
#include "cuda_dsmc/distribution_generation.hpp"
#include "cuda_dsmc/CustomSink.hpp"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    mkdir("./tmp", 0700);
    const std::string kPathToLogFile = "./tmp/";
#else
    const std::string kPathToLogFile = "/tmp/";
#endif
const std::string kLogfilename = "test_dsmc_distribution_generation";

int kNumAtoms = 1e3;
double kTestTemp = 100.e-9;
double kTestV = sqrt(kKB * kTestTemp / kMass);

double kTolerance = 5. / sqrt(kNumAtoms);

#if defined(HARMONIC)
FieldParams kTestParams = {.omega = make_double3(1., 1., 1.),
                           .B0 = 0.
                          };
double3 kTestPosStdDev = make_double3(1., 1., 1.);
#else  // No magnetic field
FieldParams kTestParams = {.B0 = 0.,
                           .max_distribution_width = 1.};
double3 kTestPosStdDev = make_double3(0.539, 0.539, 0.539);
#endif

class DistributionTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        // If using MPI get the world rank information
#if defined(MPI)
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif
        // Initialise detreministic seed
        pcg32x2_srandom_r(&rng,
                          42u,
                          42u,
                          54u,
                          54u);

        // Initialise distributions
        printf("Generating positions\n");
        generateThermalPositionDistribution(kNumAtoms,
                                            kTestParams,
                                            kTestTemp,
                                            &rng,
                                            &pos);
        generateThermalVelocityDistribution(kNumAtoms,
                                            kTestTemp,
                                            &rng,
                                            &vel);
    }

    virtual void TearDown() {
        free(pos);
        free(vel);
    }

    pcg32x2_random_t rng;

    double3 *pos, *vel;
#if defined(MPI)
    int world_size, world_rank;
#endif
};

TEST_F(DistributionTest, VelocityMean) {
    double3 directional_mean;
    double global_mean = mean(vel,
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

TEST_F(DistributionTest, VelocityStdDev) {
    double3 directional_stddev;
    double global_stddev = stddev(vel,
                                  kNumAtoms,
                                  &directional_stddev);

    ASSERT_LT(directional_stddev.x, kTestV * (1. + kTolerance));
    ASSERT_LT(directional_stddev.y, kTestV * (1. + kTolerance));
    ASSERT_LT(directional_stddev.z, kTestV * (1. + kTolerance));
    // ASSERT_LT(global_stddev, kTestV * (1. + kTolerance));

    ASSERT_GT(directional_stddev.x, kTestV * (1. - kTolerance));
    ASSERT_GT(directional_stddev.y, kTestV * (1. - kTolerance));
    ASSERT_GT(directional_stddev.z, kTestV * (1. - kTolerance));
    // ASSERT_GT(global_stddev, kTestV * (1. - kTolerance));
}

TEST_F(DistributionTest, PositionMean) {
    double3 directional_mean;
    double global_mean = mean(pos,
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

TEST_F(DistributionTest, PositionStdDev) {
    double3 directional_stddev;
    double global_stddev = stddev(pos,
                                  kNumAtoms,
                                  &directional_stddev);

    ASSERT_LT(directional_stddev.x, kTestPosStdDev.x * (1. + kTolerance));
    ASSERT_LT(directional_stddev.y, kTestPosStdDev.y * (1. + kTolerance));
    ASSERT_LT(directional_stddev.z, kTestPosStdDev.z * (1. + kTolerance));
    // ASSERT_LT(global_stddev, kTestPosStdDev * (1. + kTolerance));

    ASSERT_GT(directional_stddev.x, kTestPosStdDev.x * (1. - kTolerance));
    ASSERT_GT(directional_stddev.y, kTestPosStdDev.y * (1. - kTolerance));
    ASSERT_GT(directional_stddev.z, kTestPosStdDev.z * (1. - kTolerance));
    // ASSERT_GT(global_stddev, kTestPosStdDev * (1. - kTolerance));
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
#if defined(MPI)
    // Initialize the MPI environment
    printf("Initialising MPI\n");
    MPI_Init(&argc, &argv);
#endif

    ::testing::InitGoogleTest(&argc, argv);

    int return_value = RUN_ALL_TESTS();

#if defined(MPI)
    // Finalize the MPI environment.
    MPI_Finalize();
#endif
    return return_value;
}

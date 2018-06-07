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

double kNumAtoms = 1.e4;
double kTestTemp = 100.e-9;
double kTestV = sqrt(kKB * kTestTemp / kMass);

double kTolerance = 5. / sqrt(kNumAtoms);

#if defined(HARMONIC)
FieldParams kTestParams = {.omega = make_double3(1., 1., 1.),
                           .B0 = 0.
                          };
double3 kTestPosStdDev = make_double3(1., 1., 1.);
#else  // No magnetic field
FieldParams kTestParams = {.B0 = 0.};
double3 kTestPosStdDev = make_double3(0.539, 0.539, 0.539);
#endif

class DistributionTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
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
};

TEST_F(DistributionTest, VelocityMean) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += vel[test].x;
        test_sum.y += vel[test].y;
        test_sum.z += vel[test].z;
    }
#if defined(MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &test_sum,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
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

TEST_F(DistributionTest, VelocityStdDev) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += vel[test].x;
        test_sum.y += vel[test].y;
        test_sum.z += vel[test].z;
    }
#if defined(MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &test_sum,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    double3 test_mean = test_sum / kNumAtoms;
    double global_mean = (test_mean.x + test_mean.y + test_mean.x) / 3.;

    double3 sum_of_squared_differences = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        sum_of_squared_differences.x += (vel[test].x - test_mean.x) *
                                        (vel[test].x - test_mean.x);
        sum_of_squared_differences.y += (vel[test].y - test_mean.y) *
                                        (vel[test].y - test_mean.y);
        sum_of_squared_differences.z += (vel[test].z - test_mean.z) *
                                        (vel[test].z - test_mean.z);
    }
#if defined(MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &sum_of_squared_differences,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
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

TEST_F(DistributionTest, PositionMean) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += pos[test].x;
        test_sum.y += pos[test].y;
        test_sum.z += pos[test].z;
    }
#if defined(MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &test_sum,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
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

TEST_F(DistributionTest, PositionStdDev) {
    double3 test_sum = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        test_sum.x += pos[test].x;
        test_sum.y += pos[test].y;
        test_sum.z += pos[test].z;
    }
#if defined(MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &test_sum,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    double3 test_mean = test_sum / kNumAtoms;
    double global_mean = (test_mean.x + test_mean.y + test_mean.x) / 3.;

    double3 sum_of_squared_differences = make_double3(0., 0., 0.);
    for (int test = 0; test < kNumAtoms; ++test) {
        sum_of_squared_differences.x += (pos[test].x - test_mean.x) *
                                        (pos[test].x - test_mean.x);
        sum_of_squared_differences.y += (pos[test].y - test_mean.y) *
                                        (pos[test].y - test_mean.y);
        sum_of_squared_differences.z += (pos[test].z - test_mean.z) *
                                        (pos[test].z - test_mean.z);
    }
#if defined(MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &sum_of_squared_differences,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    double3 test_std_dev = make_double3(0., 0., 0.);
    test_std_dev.x = sqrt(sum_of_squared_differences.x / (kNumAtoms-1));
    test_std_dev.y = sqrt(sum_of_squared_differences.y / (kNumAtoms-1));
    test_std_dev.z = sqrt(sum_of_squared_differences.z / (kNumAtoms-1));
    double global_std_dev = (test_std_dev.x +
                             test_std_dev.y +
                             test_std_dev.z) / 3.;

    ASSERT_LT(test_std_dev.x, kTestPosStdDev.x * (1. + kTolerance));
    ASSERT_LT(test_std_dev.y, kTestPosStdDev.y * (1. + kTolerance));
    ASSERT_LT(test_std_dev.z, kTestPosStdDev.z * (1. + kTolerance));
    // ASSERT_LT(global_std_dev, kTestPosStdDev * (1. + kTolerance));

    ASSERT_GT(test_std_dev.x, kTestPosStdDev.x * (1. - kTolerance));
    ASSERT_GT(test_std_dev.y, kTestPosStdDev.y * (1. - kTolerance));
    ASSERT_GT(test_std_dev.z, kTestPosStdDev.z * (1. - kTolerance));
    // ASSERT_GT(global_std_dev, kTestPosStdDev * (1. - kTolerance));
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

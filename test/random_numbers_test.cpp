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

#include "declare_physical_constants.hpp"
#include "define_physical_constants.hpp"
#include "vector_math.cuh"

#include "random_numbers.hpp"
#include "CustomSink.hpp"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    mkdir("./tmp", 0700);
    const std::string kPathToLogFile = "./tmp/";
#else
    const std::string kPathToLogFile = "/tmp/";
#endif
const std::string kLogfilename = "test_dsmc_random_numbers";

int kNumberOfTests = 10000;
double kTolerance = 1. / sqrt(kNumberOfTests);

class RNGTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        // Initialise detreministic seed
        pcg32x2_srandom_r(&rng,
                          42u,
                          42u,
                          54u,
                          54u);
    }

  // virtual void TearDown() {}

    pcg32x2_random_t rng;
};

///////////////////
// UNIFORM TESTS //
///////////////////

// Test to esnure the random numbers are in the interval [0, 1)
TEST_F(RNGTest, UniformBetweenZeroAndOne) {
    double test_number = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_number = uniformRandom(&rng);
        ASSERT_LT(test_number, 1.);
        ASSERT_GE(test_number, 0.);
    }
}

TEST_F(RNGTest, UniformMean) {
    double test_sum = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += uniformRandom(&rng);
    }

    double test_mean = test_sum / kNumberOfTests;
    ASSERT_LT(test_mean, 0.5 * (1. + kTolerance));
    ASSERT_GT(test_mean, 0.5 * (1. - kTolerance));
}

TEST_F(RNGTest, UniformStdDev) {
    double random_number = 0.;
    double sum_of_squared_differences = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        random_number = uniformRandom(&rng);
        sum_of_squared_differences += (random_number - 0.5) *
                                      (random_number - 0.5);
    }

    double test_std_dev = sqrt(sum_of_squared_differences / (kNumberOfTests-1));
    ASSERT_LT(test_std_dev, sqrt(1./12.) * (1. + kTolerance));
    ASSERT_GT(test_std_dev, sqrt(1./12.) * (1. - kTolerance));
}

/////////////////////
// GAUSSIAN TESTS  //
/////////////////////

TEST_F(RNGTest, GaussianMean) {
    double2 test_rand = make_double2(0., 0.);
    double test_sum = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_rand = boxMuller(&rng);
        test_sum += test_rand.x + test_rand.y;
    }

    double test_mean = test_sum / (2*kNumberOfTests);
    ASSERT_LT(test_mean, kTolerance);
    ASSERT_GT(test_mean, -1. * kTolerance);
}

TEST_F(RNGTest, GaussianStdDev) {
    double2 random_number = make_double2(0., 0.);
    double sum_of_squared_differences = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        random_number = boxMuller(&rng);
        sum_of_squared_differences += (random_number.x - 0.0) *
                                      (random_number.x - 0.0);
        sum_of_squared_differences += (random_number.y - 0.0) *
                                      (random_number.y - 0.0);
    }

    double test_std_dev = sqrt(sum_of_squared_differences /
                              (2*kNumberOfTests-1));
    ASSERT_LT(test_std_dev, 1. * (1. + kTolerance));
    ASSERT_GT(test_std_dev, 1. * (1. - kTolerance));
}

TEST_F(RNGTest, GaussianBackOfTheEnvelope) {
    double2 random_number = make_double2(0., 0.);
    double2 test_extremum = make_double2(0., 0.);
    double test_numbers[2*kNumberOfTests];
    for (int test = 0; test < kNumberOfTests; ++test) {
        random_number = boxMuller(&rng);
        test_numbers[2*test] = random_number.x;
        test_numbers[2*test+1] = random_number.y;

        if (random_number.x > test_extremum.y)
            test_extremum.y = random_number.x;
        if (random_number.y > test_extremum.y)
            test_extremum.y = random_number.y;
        if (random_number.x < test_extremum.x)
            test_extremum.x = random_number.x;
        if (random_number.y < test_extremum.x)
            test_extremum.x = random_number.y;
    }

    double test_sum = 0.;
    for (int test = 0; test < 2*kNumberOfTests; ++test) {
        test_sum += test_numbers[test];
    }
    double test_mean = test_sum / (2*kNumberOfTests);

    double sum_of_squared_differences = 0.;
    for (int test = 0; test < 2*kNumberOfTests; ++test) {
        sum_of_squared_differences += (test_numbers[test] - test_mean) *
                                      (test_numbers[test] - test_mean);
    }
    double test_std_dev = sqrt(sum_of_squared_differences /
                              (2*kNumberOfTests-1));

    double2 z_extremum = (test_extremum - test_mean) / test_std_dev;

    if (2*kNumberOfTests < 300) {
        ASSERT_GT(z_extremum.x, -3.);
        ASSERT_LT(z_extremum.y, 3.);
    } else if (2*kNumberOfTests < 15000) {
        ASSERT_GT(z_extremum.x, -4.);
        ASSERT_LT(z_extremum.y, 4.);
    } else {
        ASSERT_GT(z_extremum.x, -5.);
        ASSERT_LT(z_extremum.y, 5.);
    }
}

/////////////////////
// VECTOR TESTS    //
/////////////////////

TEST_F(RNGTest, GaussianVector3Mean) {
    double3 test_rand = make_double3(0., 0., 0.);
    double test_sum = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_rand = gaussianVector(0.0,
                                   1.0,
                                   &rng);
        test_sum += test_rand.x + test_rand.y + test_rand.z;
    }

    double test_mean = test_sum / (3*kNumberOfTests);
    ASSERT_LT(test_mean, kTolerance);
    ASSERT_GT(test_mean, -1. * kTolerance);
}

TEST_F(RNGTest, GaussianVector3StdDev) {
    double3 random_number = make_double3(0., 0., 0.);
    double sum_of_squared_differences = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        random_number = gaussianVector(0.0,
                                       1.0,
                                       &rng);
        sum_of_squared_differences += (random_number.x - 0.0) *
                                      (random_number.x - 0.0);
        sum_of_squared_differences += (random_number.y - 0.0) *
                                      (random_number.y - 0.0);
        sum_of_squared_differences += (random_number.z - 0.0) *
                                      (random_number.z - 0.0);
    }

    double test_std_dev = sqrt(sum_of_squared_differences /
                              (3*kNumberOfTests-1));
    ASSERT_LT(test_std_dev, 1. * (1. + kTolerance));
    ASSERT_GT(test_std_dev, 1. * (1. - kTolerance));
}

TEST_F(RNGTest, GaussianVector3Distinct) {
    double3 test_rand = make_double3(0., 0., 0.);

    for (int test = 0; test < kNumberOfTests; ++test) {
        test_rand = gaussianVector(0.0,
                                   1.0,
                                   &rng);
        ASSERT_NE(test_rand.x, test_rand.y);
        ASSERT_NE(test_rand.x, test_rand.z);
        ASSERT_NE(test_rand.y, test_rand.z);
    }
}

int main(int argc, char **argv) {
    auto worker = g3::LogWorker::createLogWorker();
    auto logfileHandle = worker->addDefaultLogger(kLogfilename,
                                                  kPathToLogFile);

    // logger is initialized
    g3::initializeLogging(worker.get());

    auto stdoutHandle = worker->addSink(std2::make_unique<CustomSink>(),
                                        &CustomSink::ReceiveLogMessage);

    LOGF(INFO, "Testing random number generators.");

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

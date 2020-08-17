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

#include "gtest/gtest.h"

#include "cuda_dsmc/declare_physical_constants.hpp"
#include "cuda_dsmc/define_physical_constants.hpp"
#include "cuda_dsmc/vector_math.cuh"
#include "cuda_dsmc/random_numbers.cuh"

#include "cuda_dsmc/CustomSink.hpp"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    mkdir("./tmp", 0700);
    const std::string kPathToLogFile = "./tmp/";
#else
    const std::string kPathToLogFile = "/tmp/";
#endif
const std::string kLogfilename = "test_cuda_dsmc_random_numbers";

int kRNGSeed = 1234;
int kNumberOfTests = 1e7;
double kTolerance = 1. / sqrt(kNumberOfTests);

int kNumBlocks = 1024;
int kNumThreads = 128;

////////////////////////////
// UNIFORM HOST API TESTS //
////////////////////////////

class HostAPIUniformRNGTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        // Initialise detreministic seed
        curandGetErrorString(curandCreateGenerator(&rng, 
                                          CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, 
                                                       1234ULL));

        /* Allocate n floats on host */ 
        hostData = (double *)calloc(kNumberOfTests, sizeof(double)); 
        /* Allocate n floats on device */ 
        CUDA_CHECK(cudaMalloc((void **)&devData, kNumberOfTests*sizeof(double)));

        /* Generate n floats on device */ 
        CURAND_CHECK(curandGenerateUniformDouble(rng, devData, kNumberOfTests));
        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(hostData, devData, kNumberOfTests * sizeof(double), cudaMemcpyDeviceToHost)); 
    }

    virtual void TearDown() {
        CURAND_CHECK(curandDestroyGenerator(rng));
        CUDA_CHECK(cudaFree(devData));
        free(hostData); 
    }

    curandGenerator_t rng;

    double *devData, *hostData;
};

// Test to esnure the random numbers are in the interval [0, 1)
TEST_F(HostAPIUniformRNGTest, UniformBetweenZeroAndOne) {
    LOGF(INFO, "Checking uniform numbers lie in the interval [0,1 )");
    for (int test = 0; test < kNumberOfTests; ++test) {
        ASSERT_LT(hostData[test], 1.);
        ASSERT_GE(hostData[test], 0.);
    }
}

TEST_F(HostAPIUniformRNGTest, UniformMean) {
    LOGF(INFO, "Checking uniform numbers have a mean of 0.5 (within reason)");
    double test_sum = 0.;
    
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += hostData[test];
    }

    double test_mean = test_sum / kNumberOfTests;
    ASSERT_LT(test_mean, 0.5 * (1. + kTolerance));
    ASSERT_GT(test_mean, 0.5 * (1. - kTolerance));
}

TEST_F(HostAPIUniformRNGTest, UniformStdDev) {
    LOGF(INFO, "Checking uniform numbers have a standard deviation of sqrt(1/12) (within reason)");
    double test_sum = 0.;
    double test_mean = 0.;
    double sum_of_squared_differences = 0.;

    /* Calculate mean */
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += hostData[test];
    }
    test_mean = test_sum / kNumberOfTests;

    /* Calculate standard deviation */
    for (int test = 0; test < kNumberOfTests; ++test) {
        sum_of_squared_differences += (hostData[test] - test_mean) *
                                      (hostData[test] - test_mean);
    }

    double test_std_dev = sqrt(sum_of_squared_differences / (kNumberOfTests-1));
    ASSERT_LT(test_std_dev, sqrt(1./12.) * (1. + kTolerance));
    ASSERT_GT(test_std_dev, sqrt(1./12.) * (1. - kTolerance));
}

//////////////////////////////
// UNIFORM DEVICE API TESTS //
//////////////////////////////

__global__ void generate_uniform_kernel(curandState *state,
                                        double *result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    /* Copy state to local memory for efficiency */
    curandState l_state = state[id];
    /* Generate pseudo-random uniforms */
    result[id] = curand_uniform_double(&l_state);
    /* Copy state back to global memory */
    state[id] = l_state;

    return;
}

class DeviceAPIUniformRNGTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        /* Allocate space for rng states on device */
        CUDA_CHECK(cudaMalloc((void **)&d_states,
                             kNumBlocks * kNumThreads * sizeof(curandState)));
        /* Initialise rng states on device */
        cuInitRNG(kNumBlocks*kNumThreads,
                  kRNGSeed,
                  d_states);

        /* Allocate n floats on host */ 
        h_data = (double *)calloc(kNumBlocks * kNumThreads, sizeof(double)); 
        /* Allocate n floats on device */ 
        CUDA_CHECK(cudaMalloc((void **)&d_data, kNumBlocks * kNumThreads*sizeof(double)));

        /* Initialise rng states on device */
        generate_uniform_kernel<<<kNumBlocks,
                                  kNumThreads>>>(d_states,
                                                 d_data);
        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(h_data,
                             d_data,
                             kNumBlocks * kNumThreads * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }

    virtual void TearDown() {
        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_data));
        free(h_data);
    }

    curandState *d_states;

    double *d_data, *h_data;
};

// Test to esnure the random numbers are in the interval [0, 1)
TEST_F(DeviceAPIUniformRNGTest, UniformBetweenZeroAndOne) {
    LOGF(INFO, "Checking uniform numbers lie in the interval [0,1 )");
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        ASSERT_LT(h_data[test], 1.);
        ASSERT_GE(h_data[test], 0.);
    }
}

TEST_F(DeviceAPIUniformRNGTest, UniformMean) {
    LOGF(INFO, "Checking uniform numbers have a mean of 0.5 (within reason)");
    double test_sum = 0.;
    
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        test_sum += h_data[test];
    }

    double test_mean = test_sum / (kNumBlocks * kNumThreads);
    ASSERT_LT(test_mean, 0.5 * (1. + kTolerance));
    ASSERT_GT(test_mean, 0.5 * (1. - kTolerance));
}

TEST_F(DeviceAPIUniformRNGTest, UniformStdDev) {
    LOGF(INFO, "Checking uniform numbers have a standard deviation of sqrt(1/12) (within reason)");
    double test_sum = 0.;
    double test_mean = 0.;
    double sum_of_squared_differences = 0.;

    /* Calculate mean */
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        test_sum += h_data[test];
    }
    test_mean = test_sum / (kNumBlocks * kNumThreads);

    /* Calculate standard deviation */
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        sum_of_squared_differences += (h_data[test] - test_mean) *
                                      (h_data[test] - test_mean);
    }

    double test_std_dev = sqrt(sum_of_squared_differences / (kNumBlocks * kNumThreads-1));
    ASSERT_LT(test_std_dev, sqrt(1./12.) * (1. + kTolerance));
    ASSERT_GT(test_std_dev, sqrt(1./12.) * (1. - kTolerance));
}

//////////////////////////////
// GAUSSIAN HOST API TESTS  //
//////////////////////////////

class HostAPIGaussianRNGTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        // Initialise detreministic seed
        CURAND_CHECK(curandCreateGenerator(&rng, 
                                          CURAND_RNG_PSEUDO_DEFAULT)); 
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, 
                                                       1234ULL));

        /* Allocate n floats on host */ 
        hostData = (double *)calloc(kNumberOfTests, sizeof(double)); 
        /* Allocate n floats on device */ 
        CUDA_CHECK(cudaMalloc((void **)&devData, kNumberOfTests*sizeof(double)));

        /* Generate n floats on device */ 
        CURAND_CHECK(curandGenerateNormalDouble(rng, 
                                               devData, 
                                               kNumberOfTests, 
                                               0.0,
                                               1.0));
        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(hostData, devData, kNumberOfTests * sizeof(double), cudaMemcpyDeviceToHost)); 
    }

    virtual void TearDown() {
        CURAND_CHECK(curandDestroyGenerator(rng));
        CUDA_CHECK(cudaFree(devData));
        free(hostData); 
    }

    curandGenerator_t rng;

    double *devData, *hostData;
};

TEST_F(HostAPIGaussianRNGTest, GaussianMean) {
    LOGF(INFO, "Checking normal numbers have a mean of 0 (within reason)");
    double test_sum = 0.;

    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += hostData[test];
    }

    double test_mean = test_sum / kNumberOfTests;
    ASSERT_LT(test_mean, kTolerance);
    ASSERT_GT(test_mean, -1. * kTolerance);
}

TEST_F(HostAPIGaussianRNGTest, GaussianStdDev) {
    LOGF(INFO, "Checking normal numbers have a mean of 0 (within reason)");
    double test_sum = 0.;
    double test_mean = 0.;
    double sum_of_squared_differences = 0.;

    /* Calculate mean */
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += hostData[test];
    }
    test_mean = test_sum / kNumberOfTests;

    for (int test = 0; test < kNumberOfTests; ++test) {
        sum_of_squared_differences += (hostData[test] - test_mean) *
                                      (hostData[test] - test_mean);
    }

    double test_std_dev = sqrt(sum_of_squared_differences / (kNumberOfTests-1));
    ASSERT_LT(test_std_dev, 1. * (1. + kTolerance));
    ASSERT_GT(test_std_dev, 1. * (1. - kTolerance));
}

TEST_F(HostAPIGaussianRNGTest, GaussianBackOfTheEnvelope) {
    double2 test_extremum = make_double2(0., 0.);

    for (int test = 0; test < kNumberOfTests; ++test) {
        if (hostData[test] > test_extremum.y)
            test_extremum.y = hostData[test];
        if (hostData[test] < test_extremum.x)
            test_extremum.x = hostData[test];
    }

    double test_sum = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += hostData[test];
    }
    double test_mean = test_sum / kNumberOfTests;

    double sum_of_squared_differences = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        sum_of_squared_differences += (hostData[test] - test_mean) *
                                      (hostData[test] - test_mean);
    }
    double test_std_dev = sqrt(sum_of_squared_differences /
                              (kNumberOfTests-1));

    double2 z_extremum = (test_extremum - test_mean) / test_std_dev;

    if (kNumberOfTests < 300) {
        ASSERT_GT(z_extremum.x, -3.);
        ASSERT_LT(z_extremum.y, 3.);
    } else if (kNumberOfTests < 15000) {
        ASSERT_GT(z_extremum.x, -4.);
        ASSERT_LT(z_extremum.y, 4.);
    } else {
        ASSERT_GT(z_extremum.x, -5.);
        ASSERT_LT(z_extremum.y, 5.);
    }
}

////////////////////////////////
// GAUSSIAN DEVICE API TESTS  //
////////////////////////////////

__global__ void generate_gaussian_kernel(curandState *state,
                                         double *result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    /* Copy state to local memory for efficiency */
    curandState l_state = state[id];
    /* Generate pseudo-random uniforms */
    result[id] = curand_normal_double(&l_state);
    /* Copy state back to global memory */
    state[id] = l_state;

    return;
}

class DeviceAPIGaussianRNGTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        /* Allocate space for rng states on device */
        CUDA_CHECK(cudaMalloc((void **)&d_states,
                             kNumBlocks * kNumThreads * sizeof(curandState)));
        /* Initialise rng states on device */
        cuInitRNG(kNumBlocks*kNumThreads,
                  kRNGSeed,
                  d_states);

        /* Allocate n floats on host */ 
        h_data = (double *)calloc(kNumBlocks * kNumThreads, sizeof(double)); 
        /* Allocate n floats on device */ 
        CUDA_CHECK(cudaMalloc((void **)&d_data, kNumBlocks * kNumThreads*sizeof(double)));

        /* Initialise rng states on device */
        generate_gaussian_kernel<<<kNumBlocks, kNumThreads>>>(d_states,
                                                              d_data);
        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(h_data,
                             d_data,
                             kNumBlocks * kNumThreads * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }

    virtual void TearDown() {
        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_data));
        free(h_data);
    }

    curandState *d_states;

    double *d_data, *h_data;
};

TEST_F(DeviceAPIGaussianRNGTest, GaussianMean) {
    LOGF(INFO, "Checking normal numbers have a mean of 0 (within reason)");
    double test_sum = 0.;

    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        test_sum += h_data[test];
    }

    double test_mean = test_sum / (kNumBlocks * kNumThreads);
    ASSERT_LT(test_mean, kTolerance);
    ASSERT_GT(test_mean, -1. * kTolerance);
}

TEST_F(DeviceAPIGaussianRNGTest, GaussianStdDev) {
    LOGF(INFO, "Checking normal numbers have a mean of 0 (within reason)");
    double test_sum = 0.;
    double test_mean = 0.;
    double sum_of_squared_differences = 0.;

    /* Calculate mean */
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        test_sum += h_data[test];
    }
    test_mean = test_sum / (kNumBlocks * kNumThreads);

    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        sum_of_squared_differences += (h_data[test] - test_mean) *
                                      (h_data[test] - test_mean);
    }

    double test_std_dev = sqrt(sum_of_squared_differences / (kNumBlocks * kNumThreads-1));
    ASSERT_LT(test_std_dev, 1. * (1. + kTolerance));
    ASSERT_GT(test_std_dev, 1. * (1. - kTolerance));
}

TEST_F(DeviceAPIGaussianRNGTest, GaussianBackOfTheEnvelope) {
    double2 test_extremum = make_double2(0., 0.);

    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        if (h_data[test] > test_extremum.y)
            test_extremum.y = h_data[test];
        if (h_data[test] < test_extremum.x)
            test_extremum.x = h_data[test];
    }

    double test_sum = 0.;
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        test_sum += h_data[test];
    }
    double test_mean = test_sum / (kNumBlocks * kNumThreads);

    double sum_of_squared_differences = 0.;
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        sum_of_squared_differences += (h_data[test] - test_mean) *
                                      (h_data[test] - test_mean);
    }
    double test_std_dev = sqrt(sum_of_squared_differences /
                              (kNumBlocks * kNumThreads-1));

    double2 z_extremum = (test_extremum - test_mean) / test_std_dev;

    if (kNumBlocks * kNumThreads < 300) {
        ASSERT_GT(z_extremum.x, -3.);
        ASSERT_LT(z_extremum.y, 3.);
    } else if (kNumBlocks * kNumThreads < 15000) {
        ASSERT_GT(z_extremum.x, -4.);
        ASSERT_LT(z_extremum.y, 4.);
    } else {
        ASSERT_GT(z_extremum.x, -5.);
        ASSERT_LT(z_extremum.y, 5.);
    }
}

//////////////////////////////
// VECTOR HOST API TESTS    //
//////////////////////////////

class HostAPIVectorRNGTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        // Initialise detreministic seed
        CURAND_CHECK(curandCreateGenerator(&rng, 
                                          CURAND_RNG_PSEUDO_DEFAULT)); 
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, 
                                                       1234ULL));

        /* Allocate n floats on host */ 
        hostData = (double3 *)calloc(kNumberOfTests, sizeof(double3)); 
        /* Allocate n floats on device */ 
        CUDA_CHECK(cudaMalloc((void **)&devData, kNumberOfTests*sizeof(double3)));

        /* Generate n floats on device */ 
        CURAND_CHECK(curandGenerateNormalDouble(rng, 
                                               (double *) devData, 
                                               3*kNumberOfTests, 
                                               0.0,
                                               1.0));
        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(hostData, devData, kNumberOfTests * sizeof(double3), cudaMemcpyDeviceToHost)); 
    }

    virtual void TearDown() {
        CURAND_CHECK(curandDestroyGenerator(rng));
        CUDA_CHECK(cudaFree(devData));
        free(hostData); 
    }

    curandGenerator_t rng;

    double3 *devData, *hostData;
};

TEST_F(HostAPIVectorRNGTest, GaussianVector3Mean) {
    double test_sum = 0.;
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += hostData[test].x + hostData[test].y + hostData[test].z;
    }

    double test_mean = test_sum / (3*kNumberOfTests);
    ASSERT_LT(test_mean, kTolerance);
    ASSERT_GT(test_mean, -1. * kTolerance);
}

TEST_F(HostAPIVectorRNGTest, GaussianVector3StdDev) {
    double test_sum = 0.;
    double test_mean = 0.;
    double sum_of_squared_differences = 0.;

    /* Calculate mean */
    for (int test = 0; test < kNumberOfTests; ++test) {
        test_sum += hostData[test].x + hostData[test].y + hostData[test].z;
    }
    test_mean = test_sum / (3. * kNumberOfTests);

    /* Calculate standard deviation */
    for (int test = 0; test < kNumberOfTests; ++test) {
        sum_of_squared_differences += (hostData[test].x - test_mean) *
                                      (hostData[test].x - test_mean);
        sum_of_squared_differences += (hostData[test].y - test_mean) *
                                      (hostData[test].y - test_mean);
        sum_of_squared_differences += (hostData[test].z - test_mean) *
                                      (hostData[test].z - test_mean);
    }
    double test_std_dev = sqrt(sum_of_squared_differences /
                              (3*kNumberOfTests-1));
    ASSERT_LT(test_std_dev, 1. * (1. + kTolerance));
    ASSERT_GT(test_std_dev, 1. * (1. - kTolerance));
}

TEST_F(HostAPIVectorRNGTest, GaussianVector3Distinct) {
    for (int test = 0; test < kNumberOfTests; ++test) {
        ASSERT_NE(hostData[test].x, hostData[test].y);
        ASSERT_NE(hostData[test].x, hostData[test].z);
        ASSERT_NE(hostData[test].y, hostData[test].z);
    }
}

////////////////////////////////
// VECTOR DEVICE API TESTS    //
////////////////////////////////

__global__ void generate_gaussian_kernel(curandState *state,
                                         double3 *result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    /* Copy state to local memory for efficiency */
    curandState l_state = state[id];
    /* Generate pseudo-random uniforms */
    result[id] = (double3) dGaussianVector(0.0,
                                           1.0,
                                           &l_state);
    /* Copy state back to global memory */
    state[id] = l_state;

    return;
}

class DeviceAPIVectorRNGTest : public ::testing::Test {
 protected:
    virtual void SetUp() {
        /* Allocate space for rng states on device */
        CUDA_CHECK(cudaMalloc((void **)&d_states,
                             kNumBlocks * kNumThreads * sizeof(curandState)));
        /* Initialise rng states on device */
        cuInitRNG(kNumBlocks*kNumThreads,
                  kRNGSeed,
                  d_states);

        /* Allocate n floats on host */ 
        h_data = (double3 *)calloc(kNumBlocks * kNumThreads, sizeof(double3)); 
        /* Allocate n floats on device */ 
        CUDA_CHECK(cudaMalloc((void **)&d_data, kNumBlocks * kNumThreads*sizeof(double3)));

        /* Initialise rng states on device */
        generate_gaussian_kernel<<<kNumBlocks, kNumThreads>>>(d_states,
                                                              d_data);
        /* Copy device memory to host */ 
        CUDA_CHECK(cudaMemcpy(h_data,
                             d_data,
                             kNumBlocks * kNumThreads * sizeof(double3),
                             cudaMemcpyDeviceToHost));
    }

    virtual void TearDown() {
        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_data));
        free(h_data);
    }

    curandState *d_states;

    double3 *d_data, *h_data;
};

TEST_F(DeviceAPIVectorRNGTest, GaussianVector3Mean) {
    printf("Hello\n");
    double test_sum = 0.;
    for (int test = 0; test < kNumBlocks * kNumThreads; ++test) {
        test_sum += h_data[test].x + h_data[test].y + h_data[test].z;
    }
    printf("World\n");

    double test_mean = test_sum / (3*kNumBlocks*kNumThreads);
    ASSERT_LT(test_mean, kTolerance);
    ASSERT_GT(test_mean, -1. * kTolerance);
}

TEST_F(DeviceAPIVectorRNGTest, GaussianVector3StdDev) {
    double test_sum = 0.;
    double test_mean = 0.;
    double sum_of_squared_differences = 0.;

    /* Calculate mean */
    for (int test = 0; test < kNumBlocks*kNumThreads; ++test) {
        test_sum += h_data[test].x + h_data[test].y + h_data[test].z;
    }
    test_mean = test_sum / (3. * kNumBlocks*kNumThreads);

    /* Calculate standard deviation */
    for (int test = 0; test < kNumBlocks*kNumThreads; ++test) {
        sum_of_squared_differences += (h_data[test].x - test_mean) *
                                      (h_data[test].x - test_mean);
        sum_of_squared_differences += (h_data[test].y - test_mean) *
                                      (h_data[test].y - test_mean);
        sum_of_squared_differences += (h_data[test].z - test_mean) *
                                      (h_data[test].z - test_mean);
    }
    double test_std_dev = sqrt(sum_of_squared_differences /
                              (3*kNumBlocks*kNumThreads-1));
    ASSERT_LT(test_std_dev, 1. * (1. + kTolerance));
    ASSERT_GT(test_std_dev, 1. * (1. - kTolerance));
}

TEST_F(DeviceAPIVectorRNGTest, GaussianVector3Distinct) {
    for (int test = 0; test < kNumBlocks*kNumThreads; ++test) {
        ASSERT_NE(h_data[test].x, h_data[test].y);
        ASSERT_NE(h_data[test].x, h_data[test].z);
        ASSERT_NE(h_data[test].y, h_data[test].z);
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
#if defined(DSMC_MPI)
    // Initialize the MPI environment
    printf("Initialising MPI\n");
    MPI_Init(&argc, &argv);
    printf("... seting device id\n");
    int device_id = getLocalDeviceId();
    CUDA_CHECK(cudaSetDevice(device_id));
#endif

    ::testing::InitGoogleTest(&argc, argv);

#if defined(DSMC_MPI)
    // Finalize the MPI environment.
    MPI_Finalize();
#endif
    return RUN_ALL_TESTS();
}

/** \file
 *  \brief Random number generators and associated random kernels for the 
 *  device
 *
 *  Here we define all the kernels required for random number generation on
 *  the device, including initialisation of rngs to the generation
 *  of random vectors in cartesian space.
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "cuda_dsmc/random_numbers.cuh"

/** \brief Generates three gaussian distributed random numbers between [0, 1)
 *
 *  \param state A pointer to a cuRand rng state.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers as double3
 */
__device__ double3 dGaussianVector(double mean,
                                   double std,
                                   curandState* state) {
    double3 z = make_double3(0., 0., 0.);

    z.x = curand_normal(state);
    z.y = curand_normal(state);
    z.z = curand_normal(state);

    return mean + std * z;
}

/** \brief Generates a sample of thermally distributed velocities
 *
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers
 */
__host__ void initRNG(int num_states,
                      int seed,
                      curandState *states) {
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) gInitRNG,
                                       0,
                                       num_states);
    grid_size = (num_states + block_size - 1) / block_size;
    
    gInitRNG<<<grid_size,
               block_size>>>
            (num_states,
             seed,
             states);

    return;
}

/** \brief Initialise the random number generator on the device using the 
 *  supplied seed
 *
 *  This will initialise the state for each thread launched from the global
 *  kernel. At the moment each thread has the same seed and no offset but uses
 *  a unique sequence number (equal to its threadID). The rng scheme currently
 *  employs the curand XORWOW generator.
 *
 *  \param seed A unique seed for the rng stream.
 *  \param state A pointer to an array of cuRand rng states of the device.
 *  \exception not yet.
 */
 
__global__ void gInitRNG(int num_states,
                         int seed,
                         curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_states;
         s += blockDim.x * gridDim.x) {
        curand_init(seed, id, 0, &state[s]);
    }
}

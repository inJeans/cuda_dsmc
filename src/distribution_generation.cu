/** \file
 *  \brief Random number generators and associated random functions
 *
 *  Here we define all the functions required for random number generation,
 *  from the generation of seeds and initialisation of rngs to the generation
 *  of random vectors in cartesian space.
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "cuda_dsmc/distribution_generation.cuh"

/** \brief Generates a sample of thermally distributed positions
 *
 *  \param num_positions Number of positions to be generated (equal to the length of pos).
 *  \param temp Temperature of the thermal distribution.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \param pos A pointer to an array of double3 elements that contain the positions.
 *  \exception not yet.
 *  \return An array of thermally distributed positions
 */
__host__ void generateThermalPositionDistribution(int num_positions,
                                                  FieldParams params,
                                                  double temp,
                                                  curandState *states,
                                                  double3 **pos) {
    CUDA_CHECK(cudaMalloc((void **)&(*pos), num_positions*sizeof(double3)));

    cuGenerateThermalPositionDistribution(num_positions,
                                          params,
                                          temp,
                                          states,
                                          *pos);

    return;
}

/** \brief Generates a sample of thermally distributed positions
 *
 *  \param num_positions Number of positions to be generated (equal to the length of pos).
 *  \param temp Temperature of the thermal distribution.
 *  \param stream A CUDA stream in which to perform calculations.
 *  \param states A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \param pos A pointer to an array of double3 elements that contain the positions.
 *  \exception not yet.
 *  \return An array of thermally distributed positions
 */
__host__ void cuGenerateThermalPositionDistribution(int num_positions,
                                                    FieldParams params,
                                                    double temp,
                                                    curandState *states,
                                                    double3 *pos) {
    int block_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&grid_size,
                                       &block_size,
                                       (const void *) gGenerateThermalPosition,
                                       0,
                                       num_positions);
    
    gGenerateThermalPosition<<<grid_size,
                               block_size>>>
                            (num_positions,
                             params,
                             temp,
                             states,
                             pos);

    return;
}

/** \brief Generates a single of thermally distributed position.
 *
 *  \param temp Temperature of the thermal distribution.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers
 */
__global__ void gGenerateThermalPosition(int num_positions,
                                         FieldParams params,
                                         double temp,
                                         curandState* states,
                                         double3* pos) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState l_state = states[id]; 

    for (int p = blockIdx.x * blockDim.x + threadIdx.x;
         p < num_positions;
         p += blockDim.x * gridDim.x) {
        pos[p] = dGenerateThermalPosition(params,
                                          temp,
                                          &l_state);
    }
    states[id] = l_state;

    return;
}

/** \brief Generates a single of thermally distributed position.
 *
 *  \param temp Temperature of the thermal distribution.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers
 */
__device__ double3 dGenerateThermalPosition(FieldParams params,
                                            double temp,
                                            curandState* state) {
    bool no_atom_selected = true;
    double3 pos = make_double3(0., 0., 0.);

    while (no_atom_selected) {
        double3 r = dGaussianVector(0.,
                                    params.max_distribution_width,
                                    state);

        double mag_B = norm(dMagneticField(params,
                                           r));
        double U = 0.5 * (mag_B - params.B0) * kCuGs * kCuMuB;
        double Pr = exp(-U / kCuKB / temp);

        if (curand_uniform_double(state) < Pr) {
            pos = r;
            no_atom_selected = false;
        }
    }

    return pos;
}

/** \brief Generates a sample of thermally distributed velocities
 *
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers
 */
__host__ void generateThermalVelocityDistribution(int num_velocities,
                                                  double temp,
                                                  curandState *states,
                                                  double3 **vel) {

    CUDA_CHECK(cudaMalloc((void **)&(*vel), num_velocities*sizeof(double3)));

    cuGenerateThermalVelocityDistribution(num_velocities,
                                          temp,
                                          states,
                                          *vel);

    return;
}

/** \brief Generates a sample of thermally distributed velocities
 *
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers
 */
__host__ void cuGenerateThermalVelocityDistribution(int num_velocities,
                                                    double temp,
                                                    curandState* states,
                                                    double3 *vel) {
    int block_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&grid_size,
                                       &block_size,
                                       (const void *) gGenerateThermalVelocityDistribution,
                                       0,
                                       num_velocities);
    
    gGenerateThermalVelocityDistribution<<<grid_size,
                                           block_size>>>
                                        (num_velocities,
                                         temp,
                                         states,
                                         vel);

    return;
}

/** \brief Generates a sample of thermally distributed velocities
 *
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers
 */
__global__ void gGenerateThermalVelocityDistribution(int num_velocities,
                                                     double temp,
                                                     curandState* states,
                                                     double3 *vel) {
    double V = sqrt(kCuKB * temp / kCuMass);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState l_state = states[id]; 

    for (int v = blockIdx.x * blockDim.x + threadIdx.x;
         v < num_velocities;
         v += blockDim.x * gridDim.x) {
        vel[v] = dGaussianVector(0.0,
                                 V,
                                 &l_state);
    }
    states[id] = l_state;

    return;
}

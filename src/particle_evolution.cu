/** \file
 *  \brief Evolve distributions through time under the influence of an
 *  external magnetic field.
 *
 *  Here we define all the functions required to evolve arrays
 *  of particles through time under the influence of an external
 *  magnetic field. We are currently using a simple sympletic Euler
 *  algorithm to handle the dynamics through time.
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "particle_evolution.cuh"

/** \brief Evolves a gas of particles through time
 *
 *  \param num_particles Number of particles to be evolved (equal to the length of pos).
 *  \param params Struct containing the relevant parameters to describe the 
 *         magnetic field.
 *  \param dt Time step.
 *  \param pos A pointer to an array of double3 elements that contain the positions.
 *  \param vel A pointer to an array of double3 elements that contain the velocities.
 *  \exception not yet.
 *  \return \c void
 */
void evolveParticleDistribution(int num_particles,
                                FieldParams params,
                                double dt,
                                cudaStream_t *streams,
                                double3 ***pos,
                                double3 ***vel) {
#if defined(MPI)
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Calculate rank local number of positions
    numberElementsPerParallelUnit(world_rank,
                                  world_size,
                                  &num_particles);
#endif
    /* Get device count */
    int num_devices;
    CUDA_CALL(cudaGetDeviceCount(&num_devices));

    for (int d = 0; d < num_devices; ++d) {
        CUDA_CALL(cudaSetDevice(d));
        
        /* Allocate num_positions double3s on device */
        numberElementsPerParallelUnit(d,
                                      num_devices,
                                      &num_particles);

        cuEvolveParticleDistribution(num_particles,
                                     params,
                                     dt,
                                     streams[d],
                                     (*pos)[d],
                                     (*vel)[d]);
    }

    return;
}

/** \brief Evolves a sample of thermally distributed positions
 *
 *  \param num_particles Number of particles to be evolved (equal to the length of pos).
 *  \param params Struct containing the relevant parameters to describe the 
 *         magnetic field.
 *  \param dt Temperature of the thermal distribution.
 *  \param pos A pointer to a device array of double3 elements that 
 *         contain the positions.
 *  \param vel A pointer to a device array of double3 elements that 
 *         contain the velocities.
 *  \exception not yet.
 *  \return \c void
 */
__host__ void cuEvolveParticleDistribution(int num_particles,
                                           FieldParams params,
                                           double dt,
                                           cudaStream_t stream,
                                           double3 *pos,
                                           double3 *vel) {
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) gEvolveParticleDistribution,
                                       0,
                                       num_particles);
    grid_size = (num_particles + block_size - 1) / block_size;
    
    gEvolveParticleDistribution<<<grid_size,
                                  block_size,
                                  0,
                                  stream>>>
                               (num_particles,
                                params,
                                dt,
                                pos,
                                vel);

    return;
}

/** \brief Generates a single of thermally distributed position.
 *
 *  \param temp Temperature of the thermal distribution.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return \c void
 */
__global__ void gEvolveParticleDistribution(int num_particles,
                                            FieldParams params,
                                            double dt,
                                            double3* pos,
                                            double3* vel) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x;
         p < num_particles;
         p += blockDim.x * gridDim.x)
        dEvolveParticle(params,
                        dt,
                        &pos[p],
                        &vel[p]);

    return;
}

/** \brief Generates a single of thermally distributed position.
 *
 *  \param params Struct containg the relevant parameters to describe the 
 *         magnetic field.
 *  \param pos A pointer to a single double3 element that contains the position.
 *  \param vel A pointer to a single double3 element that contains the velocities.
 *  \exception not yet.
 *  \return \c void
 */
__device__ void dEvolveParticle(FieldParams params,
                                double dt,
                                double3 *pos,
                                double3 *vel) {
    // Symplectic Euler
    double3 acc = kCuGs * kCuMuB * dMagneticFieldGradient(params,
                                                          *pos) / kCuMass;

    vel->x += acc.x * dt;
    vel->y += acc.y * dt;
    vel->z += acc.z * dt;
    
    pos->x += vel->x * dt;
    pos->y += vel->y * dt;
    pos->z += vel->z * dt;

    return;
}


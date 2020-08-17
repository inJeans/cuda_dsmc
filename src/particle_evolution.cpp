/** \file
 *  \brief Generate thermal distributions
 *
 *  Here we define all the functions required to evolve arrays
 *  of particles through time under the influence of an external
 *  magnetic field
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "particle_evolution.hpp"

/** \brief Evolves a gas of particles through time
 *
 *  \param num_particles Number of particles to be evolved (equal to the length of pos).
 *  \param params Struct containg the relevant parameters to describe the 
 *         magnetic field.
 *  \param pos A pointer to an array of double3 elements that contain the positions.
 *  \param vel A pointer to an array of double3 elements that contain the velocities.
 *  \exception not yet.
 *  \return \c void
 */
void evolveParticleDistribution(int num_particles,
                                FieldParams params,
                                double dt,
                                double3 **pos,
                                double3 **vel) {
#if defined(DSMC_MPI)
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

    hEvolveParticleDistribution(num_particles,
                                params,
                                dt,
                                *pos,
                                *vel);

    return;
}

/** \brief Evolves a distribution particles
 *
 *  \param num_particles Number of particles to be evolved (equal to the length of pos).
 *  \param params Struct containg the relevant parameters to describe the 
 *         magnetic field.
 *  \param pos A pointer to an array of double3 elements that contain the positions.
 *  \param vel A pointer to an array of double3 elements that contain the velocities.
 *  \exception not yet.
 *  \return An array of thermally distributed positions
 */
void hEvolveParticleDistribution(int num_particles,
                                 FieldParams params,
                                 double dt,
                                 double3 *pos,
                                 double3 *vel) {
    // int nthreads = omp_get_num_threads();
    // printf("Number of threads = %d\n", nthreads);
    // #pragma omp parallel for
    for (int p = 0; p < num_particles; ++p) {
        hEvolveParticle(params,
                        dt,
                        &pos[p],
                        &vel[p]);
    }

    return;
}

/** \brief Generates a single of thermally distributed position.
 *
 *  \param params Struct containg the relevant parameters to describe the 
 *         magnetic field.
 *  \param pos A pointer to a single double3 element that contains the position.
 *  \param vel A pointer to a single double3 element that contains the velocities.
 *  \exception not yet.
 *  \return Three Gaussian distributed numbers
 */
void hEvolveParticle(FieldParams params,
                     double dt,
                     double3 *pos,
                     double3 *vel) {
    // Symplectic Euler
    double3 acc = kGs * kMuB * magneticFieldGradient(params,
                                                     *pos) / kMass;

    vel->x += acc.x * dt;
    vel->y += acc.y * dt;
    vel->z += acc.z * dt;
    
    pos->x += vel->x * dt;
    pos->y += vel->y * dt;
    pos->z += vel->z * dt;

    return;
}


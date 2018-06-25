/** \file
 *  \brief Generate thermal distributions
 *
 *  Here we define all the functions required to generate thermally
 *  distributed positions and velocities on the \b host. The generation 
 *  functions will also allocate the memory required to store the 
 *  thermal arrays.
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "distribution_generation.hpp"
#include <omp.h>

/** \brief Generates a sample of thermally distributed positions
 *
 *  \param num_positions Number of positions to be generated (equal to the length of pos).
 *  \param temp Temperature of the thermal distribution.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \param pos A pointer to an array of double3 elements that contain the positions.
 *  \exception not yet.
 *  \return \c void
 */
void generateThermalPositionDistribution(int num_positions,
                                         FieldParams params,
                                         double temp,
                                         pcg32x2_random_t* rng,
                                         double3 **pos) {
#if defined(MPI)
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Calculate rank local number of positions
    numberElementsPerRank(world_rank,
                          world_size,
                          &num_positions);
#endif
    /* Allocate num_positions double3s on host */
    *pos = reinterpret_cast<double3 *>(calloc(num_positions, sizeof(double3)));

    hGenerateThermalPositionDistribution(num_positions,
                                         params,
                                         temp,
                                         rng,
                                         *pos);

    return;
}

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
void hGenerateThermalPositionDistribution(int num_positions,
                                          FieldParams params,
                                          double temp,
                                          pcg32x2_random_t* rng,
                                          double3 *pos) {
    int nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    #pragma omp parallel for
    for (int p = 0; p < num_positions; ++p) {
        pos[p] = hGenerateThermalPosition(params,
                                          temp,
                                          rng);
    }

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
double3 hGenerateThermalPosition(FieldParams params,
                                 double temp,
                                 pcg32x2_random_t* rng) {
    bool no_atom_selected = true;
    double3 pos = make_double3(0., 0., 0.);

    while (no_atom_selected) {
        double3 r = gaussianVector(0.,
                                   kMaxDistributionWidth,
                                   rng);

        double mag_B = norm(magneticField(params,
                                          r));
        double U = 0.5 * (mag_B - params.B0) * kGs * kMuB;
        double Pr = exp(-U / kKB / temp);

        if (uniformRandom(rng) < Pr) {
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
void generateThermalVelocityDistribution(int num_velocities,
                                          double temp,
                                          pcg32x2_random_t* rng,
                                          double3 **vel) {
#if defined(MPI)
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Calculate rank local number of positions
    numberElementsPerRank(world_rank,
                          world_size,
                          &num_velocities);
#endif

    /* Allocate num_velocities double3s on host */
    *vel = reinterpret_cast<double3 *>(calloc(num_velocities, sizeof(double3)));

    hGenerateThermalVelocityDistribution(num_velocities,
                                         temp,
                                         rng,
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
void hGenerateThermalVelocityDistribution(int num_velocities,
                                          double temp,
                                          pcg32x2_random_t* rng,
                                          double3 *vel) {
    double V = sqrt(kKB * temp / kMass);

    for (int v = 0; v < num_velocities; ++v) {
        vel[v] = gaussianVector(0.0,
                                V,
                                rng);
    }

    return;
}

/** \file
 *  \brief Utility functions
 *
 *  All the common utility functions I need for doing stuff like copying,
 *  collating results from multiple devices/streams/threads, saving arrays
 *  Things of that nature
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "cuda_dsmc/dsmc_utils.hpp"

////////////////////////////////////////////////////////////////////////
//                     STATISTICAL UTILITIES                          //
////////////////////////////////////////////////////////////////////////
/** \brief Find the average of the supplied array
 *
 *  \param array Pointer to a double3 array of which we will finding the
 *         mean.
 *  \param num_elements Number of elements in the array.
 *  \param directional_mean Output variable containing the directional 
 *         means.
 *  \exception not yet.
 *  \return The global mean of the array.
 */
double mean(double3 *array,
            int num_elements,
            double3 *directional_mean) {

    double3 sum = make_double3(0., 0., 0.);
    for (int element = 0; element < num_elements; ++element) {
        sum.x += array[element].x;
        sum.y += array[element].y;
        sum.z += array[element].z;
    }
#if defined(DSMC_MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &sum,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,
                  &num_elements,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    *directional_mean = sum / num_elements;
    double global_mean = (directional_mean->x +
                          directional_mean->y +
                          directional_mean->z) / 3.;

    return global_mean;
}

/** \brief Find the standard deviation of the supplied array
 *
 *  \param array Pointer to a double3 array of which we will finding the
 *         mean.
 *  \param num_elements Number of elements in the array.
 *  \param directional_mean Output variable containing the directional 
 *         means.
 *  \exception not yet.
 *  \return The global mean of the array.
 */
double stddev(double3 *array,
              int num_elements,
              double3 *directional_stddev) {
    double3 directional_mean;
    double global_mean = mean(array,
                              num_elements,
                              &directional_mean);

    double3 sum_of_squared_differences = make_double3(0., 0., 0.);
    for (int element = 0; element < num_elements; ++element) {
        sum_of_squared_differences.x += (array[element].x - directional_mean.x) *
                                        (array[element].x - directional_mean.x);
        sum_of_squared_differences.y += (array[element].y - directional_mean.y) *
                                        (array[element].y - directional_mean.y);
        sum_of_squared_differences.z += (array[element].z - directional_mean.z) *
                                        (array[element].z - directional_mean.z);
    }
#if defined(DSMC_MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &sum_of_squared_differences,
                  3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,
                  &num_elements,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    directional_stddev->x = sqrt(sum_of_squared_differences.x / (num_elements-1));
    directional_stddev->y = sqrt(sum_of_squared_differences.y / (num_elements-1));
    directional_stddev->z = sqrt(sum_of_squared_differences.z / (num_elements-1));
    double global_std_dev = (directional_stddev->x +
                             directional_stddev->y +
                             directional_stddev->z) / 3.;

    return global_std_dev;
}

////////////////////////////////////////////////////////////////////////
//                       PHYSICAL UTILITIES                           //
////////////////////////////////////////////////////////////////////////

/** \brief Find the directional kinetic energies due to the velocity
 *
 *  \param vel Velocity object
 *  \exception not yet.
 *  \return The directional kinetic energy.
 */
double3 directionalKineticEnergy(double3 vel) {
    return 0.5 * kMass * vel * vel;
}

/** \brief Find the mean of the kinetic energy
 *
 *  \param vel Pointer to a double3 array containing velocities
 *  \param num_elements Number of elements in the array.
 *  \param directional_energy_mean Output variable containing the directional 
 *         means.
 *  \exception not yet.
 *  \return The global mean kinetic energy of the array.
 */
double kineticEnergyMean(double3 *vel,
                         int num_elements,
                         double3 *directional_energy_mean) {
    double3 directional_energy[num_elements];
    for (int particle=0; particle<num_elements; particle++) {
        directional_energy[particle] = directionalKineticEnergy(vel[particle]);
    }
#if defined(DSMC_MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &num_elements,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    double global_energy_mean = mean(directional_energy,
                                     num_elements,
                                     directional_energy_mean);

    return 3. * global_energy_mean;
}

/** \brief Find the standard deviation of the kinetic energy
 *
 *  \param vel Pointer to a double3 array containing velocities
 *  \param num_elements Number of elements in the array.
 *  \param directional_stddev Output variable containing the directional 
 *         standard deviations.
 *  \exception not yet.
 *  \return The global standard deviation of the array's kinetic energy.
 */
double kineticEnergyStddev(double3 *vel,
                           int num_elements,
                           double3 *directional_stddev) {
    double3 directional_energy[num_elements];
    for (int particle=0; particle<num_elements; particle++) {
        directional_energy[particle] = directionalKineticEnergy(vel[particle]);
    }
#if defined(DSMC_MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &num_elements,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    double global_stddev = stddev(directional_energy,
                                  num_elements,
                                  directional_stddev);
    
    return 3. * global_stddev;
}

/** \brief Find the directional potential energies due to the velocity
 *
 *  \param params Structure containing the parameters that describe the 
 *  magnetic field.
 *  \param pos Position object
 *  \exception not yet.
 *  \return The directional kinetic energy.
 */
double3 directionalPotentialEnergy(FieldParams params, 
                                   double3 pos) {
    return kGs * kMuB * magneticField(params,
                                      pos);
}

/** \brief Find the mean of the potential energy
 *
 *  \param params Structure containing the parameters that describe the 
 *  magnetic field.
 *  \param pos Pointer to a double3 array containing positions
 *  \param num_elements Number of elements in the array.
 *  \param directional_energy_mean Output variable containing the directional 
 *         means.
 *  \exception not yet.
 *  \return The global mean kinetic energy of the array.
 */
double potentialEnergyMean(double3 *pos,
                           int num_elements,
                           FieldParams params,
                           double3 *directional_energy_mean) {
    double3 directional_energy[num_elements];
    for (int particle=0; particle<num_elements; particle++) {
        directional_energy[particle] = directionalPotentialEnergy(params,
                                                                  pos[particle]);
    }
#if defined(DSMC_MPI)
    MPI_Allreduce(MPI_IN_PLACE,
                  &num_elements,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    double global_energy_mean = mean(directional_energy,
                                     num_elements,
                                     directional_energy_mean);

    return 3. * global_energy_mean;
}

/** \file
 *  \brief Unit tests for the distribution_generation file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <cuda_runtime.h>

#include <float.h>
#include <algorithm>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "random_numbers.hpp"
#include "distribution_evolution.hpp"
#include "distribution_generation.hpp"
#include "trapping_potential.hpp"
#include "distribution_evolution_tests.hpp"

#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

SCENARIO("[HOST] Acceleration Update", "[h-acc]") {
    GIVEN("A thermal distribution of 5000 positions, help in a quadrupole trap with a Bz = 2.0") {
        double init_T = 20.e-6;
        int num_test = 5000;

        // Initialise trapping parameters
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;

        // Initialise rng
        pcg32_random_t *state;
        state = reinterpret_cast<pcg32_random_t*>(calloc(num_test,
                                                         sizeof(pcg32_random_t)));

        initialise_rng_states(num_test,
                              state,
                              false);

        // Initialise positions
        double3 *pos;
        pos = reinterpret_cast<double3*>(calloc(num_test,
                                                sizeof(double3)));

        // Generate velocity distribution
        generate_thermal_positions(num_test,
                                   init_T,
                                   trap_parameters,
                                   state,
                                   pos);

        WHEN("The update_atom_accelerations function is called") {
            // Initialise accelerations
            double3 *test_acc;
            test_acc = reinterpret_cast<double3*>(calloc(num_test,
                                                         sizeof(double3)));

            // Generate accelerations
            update_atom_accelerations(num_test,
                                      trap_parameters,
                                      pos,
                                      test_acc);

            double mean_acc_x = mean_x(test_acc,
                                       num_test);
            double mean_acc_y = mean_y(test_acc,
                                       num_test);
            double mean_acc_z = mean_z(test_acc,
                                       num_test);

            double std_acc_x = std_dev_x(test_acc,
                                         num_test);
            double std_acc_y = std_dev_y(test_acc,
                                         num_test);
            double std_acc_z = std_dev_z(test_acc,
                                         num_test);

            THEN("The mean in each direction should be 0.") {
                REQUIRE(mean_acc_x <= 0. + std_acc_x / sqrt(num_test));
                REQUIRE(mean_acc_x >= 0. - std_acc_x / sqrt(num_test));
                REQUIRE(mean_acc_y <= 0. + std_acc_y / sqrt(num_test));
                REQUIRE(mean_acc_y >= 0. - std_acc_y / sqrt(num_test));
                REQUIRE(mean_acc_z <= 0. + std_acc_z / sqrt(num_test));
                REQUIRE(mean_acc_z >= 0. - std_acc_z / sqrt(num_test));
            }

            double expected_std_x_y = sqrt(trap_parameters.Bz*trap_parameters.Bz * gs*gs * muB*muB / 
                                           (48. * mass*mass));
            double expected_std_z = sqrt(trap_parameters.Bz*trap_parameters.Bz * gs*gs * muB*muB / 
                                           (12. * mass*mass));
            THEN("The standard deviation in each direction should be given by blah") {
                REQUIRE(std_acc_x <= expected_std_x_y + std_acc_x / sqrt(num_test));
                REQUIRE(std_acc_x >= expected_std_x_y - std_acc_x / sqrt(num_test));
                REQUIRE(std_acc_y <= expected_std_x_y + std_acc_y / sqrt(num_test));
                REQUIRE(std_acc_y >= expected_std_x_y - std_acc_y / sqrt(num_test));
                REQUIRE(std_acc_z <= expected_std_z + std_acc_z / sqrt(num_test));
                REQUIRE(std_acc_z >= expected_std_z - std_acc_z / sqrt(num_test));
            }

            free(test_acc);
        }

        free(pos);
    }
}

double mean_x(double3 *array,
              int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].x;

    return mean / num_elements;
}

double mean_y(double3 *array,
              int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].y;

    return mean / num_elements;
}

double mean_z(double3 *array,
              int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].z;

    return mean / num_elements;
}

double std_dev_x(double3 *array,
                 int num_elements) {
    double mu = mean_x(array,
                       num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].x-mu) * (array[i].x-mu);
    }

    return sqrt(sum / num_elements);
}

double std_dev_y(double3 *array,
                 int num_elements) {
    double mu = mean_y(array,
                       num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].y-mu) * (array[i].y-mu);
    }

    return sqrt(sum / num_elements);
}

double std_dev_z(double3 *array,
                 int num_elements) {
    double mu = mean_z(array,
                       num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].z-mu) * (array[i].z-mu);
    }

    return sqrt(sum / num_elements);
}

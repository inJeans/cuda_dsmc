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
extern "C"
{
#include "unif01.h"
#include "bbattery.h" 
}

#include "random_numbers.hpp"
#include "distribution_generation.hpp"
#include "distribution_generation_tests.hpp"

#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

SCENARIO("[HOST] Thermal velocity distribution", "[h-veldist]") {
    GIVEN("An array of appropriate seeds") {
        int num_test = 5000;

        // Initialise rng
        pcg32_random_t *state;
        state = reinterpret_cast<pcg32_random_t*>(calloc(num_test,
                                                         sizeof(pcg32_random_t)));

        initialise_rng_states(num_test,
                              state,
                              false);

        WHEN("We generate 5,000 thermal velocites with an initial temperature of 20uK") {
            double init_temp = 20.e-6;

            double3 *test_vel;
            test_vel = reinterpret_cast<double3*>(calloc(num_test,
                                                  sizeof(double3)));
            
            generate_thermal_velocities(num_test,
                                        init_temp,
                                        state,
                                        test_vel);

            printf("v0 = {%.3f, %.3f, %.3f}\n", test_vel[0].x,
                                                test_vel[0].y,
                                                test_vel[0].z);
            printf("v1 = {%.3f, %.3f, %.3f}\n", test_vel[1].x,
                                                test_vel[1].y,
                                                test_vel[1].z);

            THEN("The result give a mean speed and standard deviation as predicted by standard kinetic gas theory") {
                double speed_mean = mean_norm(test_vel,
                                              num_test);
                double speed_std = std_norm(test_vel,
                                            num_test);
                double vel_mean = mean(test_vel,
                                       num_test);
                double vel_std  = std_dev(test_vel,
                                          num_test);

                double expected_speed_mean = sqrt(8.*kB*init_temp/mass/pi);
                double expected_speed_std = sqrt((3.-8./pi)*kB*init_temp/mass);

                REQUIRE(speed_mean >= expected_speed_mean - speed_mean / sqrt(num_test));
                REQUIRE(speed_mean <= expected_speed_mean + speed_mean / sqrt(num_test));
                REQUIRE(speed_std >= expected_speed_std - speed_std / sqrt(num_test));
                REQUIRE(speed_std <= expected_speed_std + speed_std / sqrt(num_test));

                double expected_vel_mean = 0.;
                double expected_vel_std = sqrt(kB * init_temp / mass);

                REQUIRE(vel_mean >= expected_vel_mean - vel_std / sqrt(num_test));
                REQUIRE(vel_mean <= expected_vel_mean + vel_std / sqrt(num_test));
                REQUIRE(vel_std >= expected_vel_std - vel_std / sqrt(num_test));
                REQUIRE(vel_std <= expected_vel_std + vel_std / sqrt(num_test));
            }

            free(test_vel);
        }

        free(state);
    }
}

SCENARIO("[HOST] Thermal position distribution", "[h-posdist]") {
    GIVEN("An array of appropriate seeds") {
        int num_test = 5000;

        // Initialise rng
        pcg32_random_t *state;
        state = reinterpret_cast<pcg32_random_t*>(calloc(num_test,
                                                         sizeof(pcg32_random_t)));

        initialise_rng_states(num_test,
                              state,
                              false);

        WHEN("We generate 5,000 thermal positions with an initial temperature of 20uK") {
            double init_temp = 20.e-6;
            trap_geo trap_parameters;
            trap_parameters.Bz = 2.0;
            trap_parameters.B0 = 0.;

            double3 *test_pos;
            test_pos = reinterpret_cast<double3*>(calloc(num_test,
                                                  sizeof(double3)));
            
            generate_thermal_positions(num_test,
                                       init_temp,
                                       trap_parameters,
                                       state,
                                       test_pos);

            THEN("The result give a mean speed and standard deviation as predicted by standard kinetic gas theory") {
                double modified_radius_mean = mean_modified_radius(test_pos,
                                                                   num_test);
                double modified_radius_std = std_modified_radius(test_pos,
                                                                 num_test);
                double pos_mean = mean(test_pos,
                                      num_test);
                double pos_std  = std_dev(test_pos,
                                          num_test);

                double expected_radius_mean = 12.*kB*init_temp/gs/muB/trap_parameters.Bz;
                double expected_radius_std = 4.*sqrt(3)*kB*init_temp/gs/muB/trap_parameters.Bz;

                REQUIRE(modified_radius_mean >= expected_radius_mean - modified_radius_mean / sqrt(num_test));
                REQUIRE(modified_radius_mean <= expected_radius_mean + modified_radius_mean / sqrt(num_test));
                REQUIRE(modified_radius_std >= expected_radius_std - modified_radius_std / sqrt(num_test));
                REQUIRE(modified_radius_std <= expected_radius_std + modified_radius_std / sqrt(num_test));

                double expected_pos_mean = 0.;
                // double expected_pos_std = sqrt(kB * init_temp / mass);

                REQUIRE(pos_mean >= expected_pos_mean - pos_std / sqrt(num_test));
                REQUIRE(pos_mean <= expected_pos_mean + pos_std / sqrt(num_test));
                // REQUIRE(pos_std >= expected_pos_std - pos_std / sqrt(num_test));
                // REQUIRE(pos_std <= expected_pos_std + pos_std / sqrt(num_test));
            }

            free(test_pos);
        }

        free(state);
    }
}

double mean(double3 *array,
            int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += array[i].x + array[i].y + array[i].z;

    return mean / num_elements / 3.;
}

double mean_norm(double3 *array,
                 int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += norm(array[i]);

    return mean / num_elements;
}

double mean_modified_radius(double3 *pos,
                            int num_elements) {
    double mean = 0.;
    for (int i = 0; i < num_elements; ++i)
        mean += sqrt(pos[i].x*pos[i].x +
                     pos[i].y*pos[i].y + 
                     4*pos[i].z*pos[i].z);

    return mean / num_elements;
}

double std_dev(double3 *array,
               int num_elements) {
    double mu = mean(array,
                     num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i) {
        sum += (array[i].x-mu) * (array[i].x-mu);
        sum += (array[i].y-mu) * (array[i].y-mu);
        sum += (array[i].z-mu) * (array[i].z-mu);
    }

    return sqrt(sum / num_elements / 3.);
}

double std_norm(double3 *vel,
                int num_elements) {
    double mu = mean_norm(vel,
                          num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i)
        sum += (norm(vel[i])-mu) * (norm(vel[i])-mu);

    return sqrt(sum / num_elements);
}

double std_modified_radius(double3 *pos,
                           int num_elements) {
    double mu = mean_modified_radius(pos,
                                     num_elements);
    double sum = 0.;
    for (int i = 0; i < num_elements; ++i)
        sum += (sqrt(pos[i].x*pos[i].x +
                     pos[i].y*pos[i].y + 
                     4*pos[i].z*pos[i].z) - mu) * 
               (sqrt(pos[i].x*pos[i].x +
                     pos[i].y*pos[i].y + 
                     4*pos[i].z*pos[i].z) - mu);

    return sqrt(sum / num_elements);
}

double z_score(double value,
               double mean,
               double std) {
    return (value - mean) / std;
}

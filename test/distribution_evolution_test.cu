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

#include "distribution_evolution.hpp"
#include "distribution_generation.hpp"
#include "random_numbers.hpp"
#include "trapping_potential.hpp"
#include "helper_cuda.h"
#include "distribution_evolution_tests.hpp"

#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

SCENARIO("[DEVICE] Acceleration Update", "[d-acc]") {
    GIVEN("A thermal distribution of 5000 positions, help in a quadrupole trap with a Bz = 2.0") {
        int num_test = 5000;

        // Initialise trapping parameters
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;

        // Initialise rng
        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_test*sizeof(curandState)));
        initialise_rng_states(num_test,
                              state,
                              false);

        // Initialise positions
        double3 *d_test_pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_pos),
                                   num_test*sizeof(double3)));

        // Generate velocity distribution
        generate_thermal_positions(num_test,
                                   20.e-6,
                                   trap_parameters,
                                   state,
                                   d_test_pos);

        WHEN("The update_atom_accelerations function is called") {
            // Initialise accelerations
            double3 *d_test_acc;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_acc),
                                       num_test*sizeof(double3)));

            // Generate accelerations
            update_atom_accelerations(num_test,
                                      trap_parameters,
                                      d_test_pos,
                                      d_test_acc);

            THEN("The result should agree with the kinetic gas theory") {
                // REQUIRE(r >= 0.);
                // REQUIRE(r <= 1.);
            }

            cudaFree(d_test_acc);
        }

        cudaFree(d_test_pos);
    }
}

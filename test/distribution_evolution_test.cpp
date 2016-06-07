/** \file
 *  \brief Unit tests for the distribution_generation file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_evolution_tests.hpp"

double tol = 1.e-6;

SCENARIO("[HOST] Acceleration Update", "[h-acc]") {
    GIVEN("A thermal distribution of 5000 positions, help in a quadrupole trap with a Bz = 2.0") {
        double init_T = 20.e-6;
        int num_test = 5000;

        // Initialise trapping parameters
#if defined(IOFFE)  // Ioffe Pritchard trap
        trap_geo trap_parameters;
        trap_parameters.B0 = 0.01;
        trap_parameters.dB = 20.;
        trap_parameters.ddB = 40000.;
#else  // Quadrupole trap
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;
#endif

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

        // Generate position distribution
        generate_thermal_positions(num_test,
                                   init_T,
                                   trap_parameters,
                                   state,
                                   pos);

        WHEN("The update_accelerations function is called") {
            // Initialise accelerations
            double3 *test_acc;
            test_acc = reinterpret_cast<double3*>(calloc(num_test,
                                                         sizeof(double3)));

            // Initialise spins
            wavefunction *psi;
#if defined(SPIN)
            psi = reinterpret_cast<wavefunction*>(calloc(num_test,
                                                     sizeof(wavefunction)));
            generate_aligned_spins(num_test,
                                   trap_parameters,
                                   pos,
                                   psi);
#else
            psi = NULL;
#endif

            // Generate accelerations
            update_accelerations(num_test,
                                 trap_parameters,
                                 pos,
                                 test_acc,
                                 psi);

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

#if defined(IOFFE)  // Ioffe Pritchard trap
#else  // Quadrupole trap
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
#endif

            free(test_acc);
            free(psi);
        }

        free(pos);
    }
}

SCENARIO("[HOST] Velocity Update", "[h-vel]") {
    GIVEN("A thermal distribution of 5000 positions, help in a quadrupole trap with a Bz = 2.0") {
        double init_T = 20.e-6;
        int num_test = 5000;

        // Initialise trapping parameters
#if defined(IOFFE)  // Ioffe Pritchard trap
        trap_geo trap_parameters;
        trap_parameters.B0 = 0.01;
        trap_parameters.dB = 20.;
        trap_parameters.ddB = 40000.;
#else  // Quadrupole trap
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;
#endif

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

        // Generate position distribution
        generate_thermal_positions(num_test,
                                   init_T,
                                   trap_parameters,
                                   state,
                                   pos);

        // Initialise spins
        wavefunction *psi;
#if defined(SPIN)
            psi = reinterpret_cast<wavefunction*>(calloc(num_test,
                                                     sizeof(wavefunction)));
            generate_aligned_spins(num_test,
                                   trap_parameters,
                                   pos,
                                   psi);
#else
            psi = NULL;
#endif

        // Initialise accelerations
        double3 *acc;
        acc = reinterpret_cast<double3*>(calloc(num_test,
                                                sizeof(double3)));

        // Generate accelerations
        update_accelerations(num_test,
                             trap_parameters,
                             pos,
                             acc,
                             psi);

        WHEN("The update_velocities function is called with dt=1.e-6") {
            double dt = 1.e-6;
            // Initialise velocities
            double3 *test_vel;
            test_vel = reinterpret_cast<double3*>(calloc(num_test,
                                                  sizeof(double3)));

            // Generate velocity distribution
            generate_thermal_velocities(num_test,
                                        init_T,
                                        state,
                                        test_vel);

            double initial_kinetic_energy = mean_kinetic_energy(num_test,
                                                                test_vel);

            cublasHandle_t cublas_handle;
            update_velocities(num_test,
                              dt,
                              cublas_handle,
                              acc,
                              test_vel);

            double final_kinetic_energy = mean_kinetic_energy(num_test,
                                                              test_vel);

            THEN("The change in kinetic energy should be 0") {
                REQUIRE(final_kinetic_energy - initial_kinetic_energy > -tol);
                REQUIRE(final_kinetic_energy - initial_kinetic_energy < tol);
            }

            free(test_vel);
            
        }

        WHEN("The update_velocities function is called 10000 times with dt=1.e-6") {
            double dt = 1.e-6;
            // Initialise velocities
            double3 *test_vel;
            test_vel = reinterpret_cast<double3*>(calloc(num_test,
                                                  sizeof(double3)));

            // Generate velocity distribution
            generate_thermal_velocities(num_test,
                                        init_T,
                                        state,
                                        test_vel);

            double initial_kinetic_energy = mean_kinetic_energy(num_test,
                                                                test_vel);

            cublasHandle_t cublas_handle;
            for (int l = 0; l < 10000; ++l) {
                update_velocities(num_test,
                                  dt,
                                  cublas_handle,
                                  acc,
                                  test_vel);
            }

            double final_kinetic_energy = mean_kinetic_energy(num_test,
                                                              test_vel);

            THEN("The change in kinetic energy should be 0") {
                REQUIRE(final_kinetic_energy - initial_kinetic_energy > -tol);
                REQUIRE(final_kinetic_energy - initial_kinetic_energy < tol);
            }

            free(test_vel);
            
        }

        free(pos);
        free(psi);
        free(acc);
    }
}

SCENARIO("[HOST] Wavfunction Update", "[h-psiev]") {
    GIVEN("A thermal distribution of 5000 aligned atoms, help in a quadrupole trap with a Bz = 2.0") {
        double init_T = 20.e-6;
        int num_test = 5000;

        // Initialise trapping parameters
#if defined(IOFFE)  // Ioffe Pritchard trap
        trap_geo trap_parameters;
        trap_parameters.B0 = 0.01;
        trap_parameters.dB = 20.;
        trap_parameters.ddB = 40000.;
#else  // Quadrupole trap
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;
#endif

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

        // Generate position distribution
        generate_thermal_positions(num_test,
                                   init_T,
                                   trap_parameters,
                                   state,
                                   pos);

        // Initialise spins
        wavefunction *test_psi;
        test_psi = reinterpret_cast<wavefunction*>(calloc(num_test,
                                                      sizeof(wavefunction)));

        generate_aligned_spins(num_test,
                               trap_parameters,
                               pos,
                               test_psi);

        WHEN("The update_wavefunctions function is called with a dt=1.e-6") {
            double dt = 1.e-6;
            // Update wavefunctions
            update_wavefunctions(num_test,
                                 dt,
                                 trap_parameters,
                                 pos,
                                 test_psi);

            double N = 0.;
            for (int atom = 0; atom < num_test; ++atom) {
                cuDoubleComplex N2 = cuConj(test_psi[atom].up) * test_psi[atom].up + 
                                     cuConj(test_psi[atom].dn) * test_psi[atom].dn;
                N += sqrt(N2.x);
            }
            N /= num_test;

            THEN("Unitarity of the system should be maintained") {
                REQUIRE(N < 1. + tol);
                REQUIRE(N > 1. - tol);
            }

            free(test_psi);
        }

        WHEN("The update_wavefunctions function is called 10000 times with a dt=1.e-6") {
            double dt = 1.e-6;
            // Update wavefunctions
            for (int l = 0; l < 10000; ++l) {
                update_wavefunctions(num_test,
                                     dt,
                                     trap_parameters,
                                     pos,
                                     test_psi);
            }

            double N = 0.;
            for (int atom = 0; atom < num_test; ++atom) {
                cuDoubleComplex N2 = cuConj(test_psi[atom].up) * test_psi[atom].up + 
                                     cuConj(test_psi[atom].dn) * test_psi[atom].dn;
                N += sqrt(N2.x);
            }
            N /= num_test;

            THEN("Unitarity of the system should be maintained") {
                REQUIRE(N < 1. + tol);
                REQUIRE(N > 1. - tol);
            }

            free(test_psi);
        }

        free(pos);
    }
}

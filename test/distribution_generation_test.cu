/** \file
 *  \brief Unit tests for the distribution_generation file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */


#include "distribution_generation_tests.cuh"

double tol = 1.e-6;

SCENARIO("[DEVICE] Thermal velocity distribution", "[d-veldist]") {
    GIVEN("An array of appropriate seeds") {
        int num_test = 5000;

        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_test*sizeof(curandState)));
        initialise_rng_states(num_test,
                              state);

        WHEN("We generate 5,000 thermal velocites with an initial temperature of 20uK") {
            double init_temp = 20.e-6;

            double3 *d_test_vel;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_vel),
                                       num_test*sizeof(double3)));
            
            generate_thermal_velocities(num_test,
                                        init_temp,
                                        state,
                                        d_test_vel);

            double3 *test_vel;
            test_vel = reinterpret_cast<double3*>(calloc(num_test,
                                                 sizeof(double3)));
            checkCudaErrors(cudaMemcpy(test_vel,
                                       d_test_vel,
                                       num_test*sizeof(double3),
                                       cudaMemcpyDeviceToHost));

            THEN("The result give a mean speed and standard deviation as predicted by standard kinetic gas theory") {
                double speed_mean = mean_norm(test_vel,
                                               num_test);
                double speed_std = std_norm(test_vel,
                                             num_test);
                double vel_mean = mean(test_vel,
                                      num_test);
                double vel_std  = std_dev(test_vel,
                                          num_test);

                double expected_speed_mean = sqrt(8*kB*init_temp/mass/h_pi);
                double expected_speed_std = sqrt((3-8/h_pi)*kB*init_temp/mass);

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

            cudaFree(d_test_vel);
            free(test_vel);
        }

        cudaFree(state);
    }
}

SCENARIO("[DEVICE] Thermal position distribution", "[d-posdist]") {
    GIVEN("An array of appropriate seeds") {
        int num_test = 5000;

        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_test*sizeof(curandState)));
        initialise_rng_states(num_test,
                              state);

        WHEN("We generate 5,000 thermal positions with an initial temperature of 20uK") {
            double init_temp = 20.e-6;
            trap_geo trap_parameters;
            trap_parameters.Bz = 2.0;
            trap_parameters.B0 = 0.;

            double3 *d_test_pos;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_pos),
                                       num_test*sizeof(double3)));
            
            generate_thermal_positions(num_test,
                                       init_temp,
                                       trap_parameters,
                                       state,
                                       d_test_pos);

            double3 *test_pos;
            test_pos = reinterpret_cast<double3*>(calloc(num_test,
                                                 sizeof(double3)));
            checkCudaErrors(cudaMemcpy(test_pos,
                                       d_test_pos,
                                       num_test*sizeof(double3),
                                       cudaMemcpyDeviceToHost));

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

            cudaFree(d_test_pos);
            free(test_pos);
        }

        cudaFree(state);
    }
}

SCENARIO("[DEVICE] Wavefunction generation", "[d-psigen]") {
    GIVEN("A thermal distribution of positions") {
        int num_test = 5000;

        // Initialise rng
        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_test*sizeof(curandState)));
        initialise_rng_states(num_test,
                              state);

        double init_temp = 20.e-6;
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;

        double3 *d_pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pos),
                                   num_test*sizeof(double3)));

        generate_thermal_positions(num_test,
                                   init_temp,
                                   trap_parameters,
                                   state,
                                   d_pos);

        WHEN("We generate the corresponding locally aligned spins") {
            zomplex2 *d_test_psi;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_psi),
                                       num_test*sizeof(zomplex2)));

            generate_aligned_spins(num_test,
                                   trap_parameters,
                                   d_pos,
                                   d_test_psi);

            double3 *pos;
            pos = reinterpret_cast<double3*>(calloc(num_test,
                                                    sizeof(double3)));
            checkCudaErrors(cudaMemcpy(pos,
                                       d_pos,
                                       num_test*sizeof(double3),
                                       cudaMemcpyDeviceToHost));

            zomplex2 *test_psi;
            test_psi = reinterpret_cast<zomplex2*>(calloc(num_test,
                                                   sizeof(zomplex2)));
            checkCudaErrors(cudaMemcpy(test_psi,
                                       d_test_psi,
                                       num_test*sizeof(zomplex2),
                                       cudaMemcpyDeviceToHost));

            cuDoubleComplex P = make_cuDoubleComplex(0., 0.);
            for (int atom = 0; atom < num_test; ++atom) {
                double3 Bn = unit(B(pos[atom],
                                trap_parameters));
                P = P + project(Bn,
                                test_psi[atom]);
            }
            P = P / num_test;

            THEN("The mean projection onto the local magnetic field should be real and equal to 1.") {
                REQUIRE(P.x < 1. + tol);
                REQUIRE(P.x > 1. - tol);
                REQUIRE(P.y < 0. + tol);
                REQUIRE(P.y > 0. - tol);
            }

            double N = 0.;
            for (int atom = 0; atom < num_test; ++atom) {
                cuDoubleComplex N2 = cuConj(test_psi[atom].up) * test_psi[atom].up + 
                                     cuConj(test_psi[atom].dn) * test_psi[atom].dn;
                N += sqrt(N2.x);
            }
            N /= num_test;

            THEN("The mean norm of the wavefunction should be equal to 1.") {
                REQUIRE(N < 1. + tol);
                REQUIRE(N > 1. - tol);
            }


            cudaFree(d_test_psi);
            free(pos);
            free(test_psi);
        }

        cudaFree(state);
        cudaFree(d_pos);
    }
}

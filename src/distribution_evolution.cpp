/** \file
 *  \brief Functions necessary for evolvigng a distribution of atoms
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_evolution.hpp"
#ifdef CUDA
#include "distribution_evolution.cuh"
#endif

#include "declare_host_constants.hpp"

void velocity_verlet_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            cublasHandle_t cublas_handle,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc,
                            wavefunction *psi) {
    update_velocities(num_atoms,
                      0.5*dt,
                      cublas_handle,
                      acc,
                      vel);
#if defined(SPIN)
    // Record spin projections
    double up_pop[num_atoms];
    double dn_pop[num_atoms];
    project_wavefunctions(num_atoms,
                          params,
                          pos,
                          psi,
                          up_pop,
                          dn_pop);

    update_wavefunctions(num_atoms,
                         dt,
                         params,
                         pos,
                         psi);
#endif // Spin
    update_positions(num_atoms,
                     dt,
                     cublas_handle,
                     vel,
                     pos);
#if defined(SPIN)
    // Perform flip
    // Exponential Decay
    // Normalisation
#endif // Spin
    update_accelerations(num_atoms,
                         params,
                         pos,
                         acc,
                         psi);
    update_velocities(num_atoms,
                      0.5*dt,
                      cublas_handle,
                      acc,
                      vel);
    return;
}

void sympletic_euler_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            cublasHandle_t cublas_handle,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc,
                            wavefunction *psi) {
    update_velocities(num_atoms,
                      dt,
                      cublas_handle,
                      acc,
                      vel);
#if defined(SPIN)
    update_wavefunctions(num_atoms,
                         dt,
                         params,
                         pos,
                         psi);
#endif
    update_positions(num_atoms,
                     dt,
                     cublas_handle,
                     vel,
                     pos);
    update_accelerations(num_atoms,
                         params,
                         pos,
                         acc,
                         psi);

    return;
}

/** \fn void update_positions(int num_atoms,
 *                            double dt,
 *                            double3 *vel,
 *                            double3 *pos)  
 *  \brief Calls the function to update a `double3` host or device array with
 *  positions based on the atoms position, velocity and given time step.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param dt Length of the time step (seconds).
 *  \param *vel Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the velocities.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \exception not yet.
 *  \return void
*/

void update_positions(int num_atoms,
                      double dt,
                      cublasHandle_t cublas_handle,
                      double3 *vel,
                      double3 *pos) {
#if defined(CUDA)
    cu_update_positions(num_atoms,
                        dt,
                        cublas_handle,
                        vel,
                        pos);
#elif defined(MKL)
    cblas_daxpy(3*num_atoms,
                dt,
                reinterpret_cast<double *>(vel),
                1,
                reinterpret_cast<double *>(pos),
                1);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        pos[atom] = update_atom_position(dt,
                                         pos[atom],
                                         vel[atom]);
    }
#endif

    return;
}

double3 update_atom_position(double dt,
                             double3 pos,
                             double3 vel) {
    return pos + vel * dt;
}

/** \fn void update_velocities(int num_atoms,
 *                             double dt,
 *                             double3 *acc,
 *                             double3 *vel) 
 *  \brief Calls the function to update a `double3` host or device array with
 *  velocities based on the atoms velocity, acceleration and given time step.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param dt Length of the time step (seconds).
 *  \param *acc Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the accelerations.
 *  \param *vel Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the velocities.
 *  \exception not yet.
 *  \return void
*/

void update_velocities(int num_atoms,
                       double dt,
                       cublasHandle_t cublas_handle,
                       double3 *acc,
                       double3 *vel) {
#if defined(CUDA)
    cu_update_velocities(num_atoms,
                         dt,
                         cublas_handle,
                         acc,
                         vel);
#elif defined(MKL)
    cblas_daxpy(3*num_atoms,
                dt,
                reinterpret_cast<double *>(acc),
                1,
                reinterpret_cast<double *>(vel),
                1);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        vel[atom] = update_atom_velocity(dt,
                                         vel[atom],
                                         acc[atom]);
    }
#endif

    return;
}

double3 update_atom_velocity(double dt,
                             double3 vel,
                             double3 acc) {
    return vel + acc * dt;
}

/** \fn void update_accelerations(int num_atoms,
 *                                trap_geo params,
 *                                double3 *pos,
 *                                double3 *acc,
 *                                wavefunction *psi) 
 *  \brief Calls the function to update a `double3` host or device array with
 *  accelerations based on the atoms position and the trapping potential.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \param *acc Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the accelerations.
 *  \param *psi Pointer to a `wavefunction` host or device array of length
 *  `num_atoms` containing the atoms' wavefunctions.
 *  \exception not yet.
 *  \return void
*/

void update_accelerations(int num_atoms,
                          trap_geo params,
                          double3 *pos,
                          double3 *acc,
                          wavefunction *psi) {
#ifdef CUDA
    cu_update_accelerations(num_atoms,
                            params,
                            pos,
                            psi,
                            acc);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
#if defined(SPIN)
        acc[atom] = update_atom_acceleration(params,
                                             pos[atom],
                                             psi[atom]);
#else
        acc[atom] = update_atom_acceleration(params,
                                             pos[atom]);
#endif // SPIN
    }
#endif // CUDA
    return;
}

double3 update_atom_acceleration(trap_geo params,
                                 double3 pos) {
    double3 acc = make_double3(0., 0., 0.);

    acc.x = dV_dx(pos,
                  params) / mass;
    acc.y = dV_dy(pos,
                  params) / mass;
    acc.z = dV_dz(pos,
                  params) / mass;

    return acc;
}

double3 update_atom_acceleration(trap_geo params,
                                 double3 pos,
                                 wavefunction psi) {
    double3 acc = make_double3(0., 0., 0.);

    acc.x = expectation_dV_dx(params,
                              pos,
                              psi) / mass;
    acc.y = expectation_dV_dy(params,
                              pos,
                              psi) / mass;
    acc.z = expectation_dV_dz(params,
                              pos,
                              psi) / mass;
    return acc;
}

/** \fn void update_wavefunctions(int num_atoms,
 *                                double dt,
 *                                trap_geo params,
 *                                double3 *pos,
 *                                wavefunction *psi) 
 *  \brief TODO.
 *  \param num_atoms Number of atoms in the array.
 *  \param dt Size of the timestep over which to evolve.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos Pointer to a `double3` host or device array of length
 *  `num_atoms` containing the positions.
 *  \param *psi Pointer to a `wavefunction` host or device array of length
 *  `num_atoms` containing the wavefunctions.
 *  \exception not yet.
 *  \return void
*/

void update_wavefunctions(int num_atoms,
                          double dt,
                          trap_geo params,
                          double3 *pos,
                          wavefunction *psi) {
#ifdef CUDA
    cu_update_wavefunctions(num_atoms,
                            dt,
                            params,
                            pos,
                            psi);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        psi[atom] = update_atom_wavefunction(dt,
                                             params,
                                             pos[atom],
                                             psi[atom]);
    }
#endif

    return;
}

wavefunction update_atom_wavefunction(double dt,
                                      trap_geo params,
                                      double3 pos,
                                      wavefunction psi) {
    double3 mag_field = B(pos,
                          params);
    double3 Bn = unit(mag_field);
    double norm_B = norm(mag_field);

    double delta_theta = 0.5*gs*muB*norm_B*dt / hbar;
    double cos_delta_theta = cos(delta_theta);
    double sin_delta_theta = sin(delta_theta);

    cuDoubleComplex U[2][2] = {make_cuDoubleComplex(0., 0.)};
    U[0][0] = make_cuDoubleComplex(cos_delta_theta,
                                   -Bn.z*sin_delta_theta);
    U[0][1] = make_cuDoubleComplex(-Bn.y*sin_delta_theta,
                                   -Bn.x*sin_delta_theta);
    U[1][0] = make_cuDoubleComplex(Bn.y*sin_delta_theta,
                                   -Bn.x*sin_delta_theta);
    U[1][1] = make_cuDoubleComplex(cos_delta_theta,
                                   Bn.z*sin_delta_theta);

    wavefunction updated_psi = make_wavefunction(0., 0., 0., 0., psi.isSpinUp);
    updated_psi.up = U[0][0]*psi.up + U[0][1]*psi.dn;
    updated_psi.dn = U[1][0]*psi.up + U[1][1]*psi.dn;

    return updated_psi;
}

void project_wavefunctions(int num_atoms,
                           trap_geo params,
                           double3 *pos,
                           wavefunction *psi,
                           double *up_pop,
                           double *dn_pop) {
#if defined(CUDA)
    // cu_project_wavefunctions(num_atoms,
    //                          params,
    //                          pos,
    //                          psi,
    //                          up_pop,
    //                          dn_pop);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        double3 Bn = unit(B(pos[atom],
                            params));
        up_pop[atom] = cuCreal(project_up(Bn,
                                          psi[atom]));
        dn_pop[atom] = cuCreal(project_dn(Bn,
                                          psi[atom]));
    }
#endif

    return;
}

cuDoubleComplex project_up(double3 Bn,
                           wavefunction psi) {
    cuDoubleComplex P = make_cuDoubleComplex(0., 0.);
    P = 0.5 * (((1.-Bn.z)*psi.dn + Bn.x*psi.up)*cuConj(psi.dn) +
               ((1.+Bn.z)*psi.up + Bn.x*psi.dn)*cuConj(psi.up)) -
            Bn.y*cuCimag(psi.up*cuConj(psi.dn));

    return P;
}

cuDoubleComplex project_dn(double3 Bn,
                           wavefunction psi) {
    cuDoubleComplex P = make_cuDoubleComplex(0., 0.);
    P = 0.5 * (((1.+Bn.z)*psi.dn - Bn.x*psi.up)*cuConj(psi.dn) +
               ((1.-Bn.z)*psi.up - Bn.x*psi.dn)*cuConj(psi.up)) +
            Bn.y*cuCimag(psi.up*cuConj(psi.dn));

    return P;
}

void flip_wavefunctions(int num_atoms,
                        trap_geo params,
                        double3 *pos,
                        double3 *vel,
                        double *up_pop,
                        double *dn_pop,
                        wavefunction *psi,
                        pcg32_random_t *state) {
    for (int atom = 0; atom < num_atoms; ++atom) {
        double3 Bn = unit(B(pos[atom],
                            params));
        double new_up_pop = cuCreal(project_up(Bn,
                                               psi[atom]));
        double new_dn_pop = cuCreal(project_dn(Bn,
                                               psi[atom]));

        double probability_of_flip = 0.;
        if (psi[atom].isSpinUp) 
            probability_of_flip = (new_dn_pop - dn_pop[atom]) / new_up_pop;
        else
            probability_of_flip = (new_up_pop - up_pop[atom]) / new_dn_pop;

        if (uniform_prng(&state[atom]) < probability_of_flip)
            psi[atom].isSpinUp = !psi[atom].isSpinUp;
    }

    return;
}

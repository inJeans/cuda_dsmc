/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "distribution_generation.hpp"
#ifdef CUDA
#include "distribution_generation.cuh"
#endif

#include "declare_host_constants.hpp"

const double max_grid_width = 2.e-3;

/* \fn void generate_aligned_spins(int num_atoms,
 *                                 trap_geo params,
 *                                 double3 *pos,
 *                                 wavefunction *psi) 
 *  \brief Calls the function to fill a `zomplex2` array of aligned spins 
 *  on the host or device with a mean projection of 1.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param params (TODO).
 *  \param *pos Pointer to a `double3` host or device array of length `num_atoms`.
 *  \param *psi Pointer to a `wavefunction` host or device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

void generate_aligned_spins(int num_atoms,
                            trap_geo params,
                            double3 *pos,
                            wavefunction *psi) {
#if defined(CUDA)
    cu_generate_aligned_spins(num_atoms,
                              params,
                              pos,
                              psi);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        psi[atom] = aligned_wavefunction(params,
                                         pos[atom]);
    }
#endif
    return;
}

wavefunction aligned_wavefunction(trap_geo params,
                                  double3 pos) {
    wavefunction aligned_psi = make_wavefunction(0., 0., 0., 0., true);

    aligned_psi = aligned_spin(params,
                               pos);

    return aligned_psi;
}

wavefunction aligned_spin(trap_geo params,
                      double3 pos) {
    wavefunction psi = make_wavefunction(0., 0., 0., 0., true);

    double3 Bn = unit(B(pos,
                        params));

    psi.up.x = 0.5 * (1. + Bn.x + Bn.z) / sqrt(1 + Bn.x);
    psi.up.y = 0.5 * (-1.*Bn.y) / sqrt(1 + Bn.x);
    psi.dn.x = 0.5 * (1. + Bn.x - Bn.z) / sqrt(1 + Bn.x);
    psi.dn.y = 0.5 * (Bn.y) / sqrt(1 + Bn.x);

    return psi;
}

/** \fn void generate_thermal_velocities(int num_atoms,
 *                                       double temp,
 *                                       curandState *state,
                                         double3 *vel) 
 *  \brief Calls the function to fill a `double3` array of thermal velocties 
 *  on the device with a mean temperature of `temp`.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param *state Pointer to a `curandState` device array of length `num_atoms`.
 *  \param *vel Pointer to a `double3` device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

#if defined(CUDA)
void generate_thermal_velocities(int num_atoms,
                                 double temp,
                                 curandState *state,
                                 double3 *vel) {
    cu_generate_thermal_velocities(num_atoms,
                                   temp,
                                   state,
                                   vel);

    return;
}
#endif

/** \fn void generate_thermal_velocities(int num_atoms,
 *                                       double temp,
 *                                       pcg32_random_t *state,
 *                                       double3 *vel)
 *  \brief Calls the function to fill a `double3` array of thermal velocties 
 *  on the host with a mean temperature of `temp`.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param *state Pointer to a `pcg32_random_t` host array of length `num_atoms`.
 *  \param *vel Pointer to a `double3` host array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

void generate_thermal_velocities(int num_atoms,
                                 double temp,
                                 pcg32_random_t *state,
                                 double3 *vel) {
    for (int atom = 0; atom < num_atoms; ++atom) {
        vel[atom] = thermal_vel(temp,
                                &state[atom]);
    }

    return;
}

/** \fn thermal_vel(double temp,
                      pcg32_random_t *state)
 *  \brief Calls the function to generate a `double3` thermal velocty on the
 *  host with a mean temperature of `temp`.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param *state Pointer to a single `pcg32_random_t` state on the host.
 *  \exception not yet.
 *  \return void
*/

double3 thermal_vel(double temp,
                    pcg32_random_t *state) {
    double V = sqrt(kB * temp / mass);
    double3 vel = gaussian_point(0,
                                 V,
                                 state);
    return vel;
}

/** \fn void generate_thermal_positions(int num_atoms,
 *                                      double temp,
 *                                      trap_geo params,
 *                                      curandState *state,
 *                                      double3 *pos)
 *  \brief Calls the function to fill a `double3` array of thermal positions 
 *  on the device with a distribution determined by the trapping potential.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param params TODO
 *  \param *state Pointer to a `curandState` device array of length `num_atoms`.
 *  \param *pos Pointer to a `double3` device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/
#ifdef CUDA
void generate_thermal_positions(int num_atoms,
                                double temp,
                                trap_geo params,
                                curandState *state,
                                double3 *pos) {
    cu_generate_thermal_positions(num_atoms,
                                  temp,
                                  params,
                                  state,
                                  pos);

    return;
}
#endif

/** \fn void generate_thermal_positions(int num_atoms,
 *                                      double temp,
 *                                      trap_geo params,
 *                                      pcg32_random_t *state,
 *                                      double3 *pos)
 *  \brief Calls the function to fill a `double3` array of thermal positions 
 *  on the host with a distribution determined by the trapping potential.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param params TODO
 *  \param *state Pointer to a `pcg32_random_t` host array of length `num_atoms`.
 *  \param *pos Pointer to a `double3` host array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

void generate_thermal_positions(int num_atoms,
                                double temp,
                                trap_geo params,
                                pcg32_random_t *state,
                                double3 *pos) {
    for (int atom = 0; atom < num_atoms; ++atom) {
        pos[atom] = thermal_pos(temp,
                                params,
                                &state[atom]);
    }

    return;
}

/** \fn thermal_pos(double temp,
 *                  trap_geo params,
 *                  pcg32_random_t *state)
 *  \brief Calls the function to generate a `double3` thermal pos on the
 *  host with a distribution determined by the trapping potential.
 *  \param temp Mean temperature of thermal gas, as defined by (TODO).
 *  \param params TODO
 *  \param *state Pointer to a single `pcg32_random_t` state on the host.
 *  \exception not yet.
 *  \return void
*/

double3 thermal_pos(double temp,
                    trap_geo params,
                    pcg32_random_t *state) {
    bool no_atom_selected = true;
    double3 pos = make_double3(0., 0., 0.);

    while (no_atom_selected) {
        // double3 r = gaussian_point(0.,
        //                            1.,
        //                            state);
        double3 r = make_double3(0., 0., 0.);
        r.x = 2. * uniform_prng(state) - 1.;
        r.y = 2. * uniform_prng(state) - 1.;
        r.z = 2. * uniform_prng(state) - 1.;
        r = r * max_grid_width;

        double magB = norm(B(r,
                             params));
        double U = 0.5 * (magB - params.B0) * gs * muB;
        double Pr = exp(-U / kB / temp);

        if (uniform_prng(state) < Pr) {
            pos = r;
            no_atom_selected = false;
        }
    }

    return pos;
}

/** \fn void initialise_atom_id(int num_atoms,
 *                              int *atom_id)
 *  \brief Calls the function to fill an `int` array of atom_ids 
 *  on the host or device.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param *atom_id Pointer to an `int` host or device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

void initialise_atom_id(int num_atoms,
                        int *atom_id) {
#if defined(CUDA)
    cu_initialise_atom_id(num_atoms,
                          atom_id);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        atom_id[atom] = atom;
    }
#endif
    return;
}

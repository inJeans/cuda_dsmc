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

/** \fn void update_atom_accelerations(int num_atoms,
 *                                     trap_geo params,
 *                                     double3 *pos,
 *                                     double3 *acc) 
 *  \brief Calls the device function to fill a `double3` array with accelerations 
 *  based on the atoms position and the trapping potential.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param params Customized structure of type `trap_geo` containing the 
 *  necessary constants for describing the trapping potential.
 *  \param *pos Pointer to a `double3` device array of length `num_atoms`.
 *  \param *acc Pointer to a `double3` device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
*/

void update_atom_accelerations(int num_atoms,
                               trap_geo params,
                               double3 *pos,
                               double3 *acc) {
#ifdef CUDA
    cu_update_atom_accelerations(num_atoms,
                                 params,
                                 pos,
                                 acc);
#else
    for (int atom = 0; atom < num_atoms; ++atom) {
        acc[atom] = update_acceleration(pos[atom],
                                        params);
    }
#endif

    return;
}

double3 update_acceleration(double3 pos,
                            trap_geo params) {
    double3 acc = make_double3(0., 0., 0.);

    acc.x = dV_dx(pos,
                  params) / mass;
    acc.y = dV_dy(pos,
                  params) / mass;
    acc.z = dV_dz(pos,
                  params) / mass;

    return acc;
}

/** \file
 *  \brief Definition of the trapping potential
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "declare_host_constants.hpp"
#include "trapping_potential.hpp"

__host__ double dV_dx(double3 pos,
                      trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * muB * gs * dot(unit_B,
                                 dB_dx(pos,
                                       params));
}

__host__ double dV_dy(double3 pos,
                      trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * muB * gs * dot(unit_B,
                                 dB_dy(pos,
                                       params));
}

__host__ double dV_dz(double3 pos,
                      trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * muB * gs * dot(unit_B,
                                 dB_dz(pos,
                                       params));
}

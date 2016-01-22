/** \file
 *  \brief Definition of the trapping potential
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "define_device_constants.cuh"
#include "trapping_potential.cuh"

__host__ __device__ double3 B(double3 pos,
                               trap_geo params) {
    double3 mag_field = make_double3(0., 0., 0.);

    mag_field.x = 0.5 * params.Bz * pos.x;
    mag_field.y = 0.5 * params.Bz * pos.y;
    mag_field.z =-1.0 * params.Bz * pos.z;

    return mag_field;
}

__host__ __device__ double3 dB_dx(double3 pos,
                                  trap_geo params) {
    double3 dBdx = make_double3(0.5 * params.Bz,
                                0.,
                                0.);

    return dBdx;
}

__host__ __device__ double3 dB_dy(double3 pos,
                                  trap_geo params) {
    double3 dBdy = make_double3(0.,
                                0.5 * params.Bz,
                                0.);

    return dBdy;
}

__host__ __device__ double3 dB_dz(double3 pos,
                                  trap_geo params) {
    double3 dBdz = make_double3(0.,
                                0.,
                                -1.0 * params.Bz);

    return dBdz;
}

__device__ double d_dV_dx(double3 pos,
                          trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * d_muB * d_gs * dot(unit_B,
                                     dB_dx(pos,
                                           params));
}

__device__ double d_expectation_dV_dx(trap_geo params,
                                      double3 pos,
                                      zomplex2 psi) {
    double3 dBdx = dB_dx(pos,
                         params);
    // Create differentiated potential operator
    cuDoubleComplex dVdx[2][2] = {make_cuDoubleComplex(0., 0.)};
    dVdx[0][0] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdx.z,
                                                       0.);
    dVdx[0][1] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdx.x,
                                                       -dBdx.y);
    dVdx[1][0] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdx.x,
                                                       dBdx.y);
    dVdx[1][1] = 0.5*d_muB*d_gs * make_cuDoubleComplex(-dBdx.z,
                                                       0.);
    // Perform expectation value calculation
    cuDoubleComplex E_dVdx = cuConj(psi.up) * (psi.up*dVdx[0][0] + psi.dn*dVdx[0][1]) + 
                             cuConj(psi.dn) * (psi.up*dVdx[1][0] + psi.dn*dVdx[1][1]);
    return cuCreal(E_dVdx);
}

__device__ double d_dV_dy(double3 pos,
                          trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * d_muB * d_gs * dot(unit_B,
                                     dB_dy(pos,
                                           params));
}

__device__ double d_expectation_dV_dy(trap_geo params,
                                      double3 pos,
                                      zomplex2 psi) {
    double3 dBdy = dB_dy(pos,
                         params);
    // Create differentiated potential operator
    cuDoubleComplex dVdy[2][2] = {make_cuDoubleComplex(0., 0.)};
    dVdy[0][0] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdy.z,
                                                       0.);
    dVdy[0][1] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdy.x,
                                                       -dBdy.y);
    dVdy[1][0] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdy.x,
                                                       dBdy.y);
    dVdy[1][1] = 0.5*d_muB*d_gs * make_cuDoubleComplex(-dBdy.z,
                                                       0.);
    // Perform expectation value calculation
    cuDoubleComplex E_dVdy = cuConj(psi.up) * (psi.up*dVdy[0][0] + psi.dn*dVdy[0][1]) + 
                             cuConj(psi.dn) * (psi.up*dVdy[1][0] + psi.dn*dVdy[1][1]);
    return cuCreal(E_dVdy);
}

__device__ double d_dV_dz(double3 pos,
                          trap_geo params) {
    double3 unit_B = unit(B(pos,
                            params));
    return -0.5 * d_muB * d_gs * dot(unit_B,
                                     dB_dz(pos,
                                           params));
}

__device__ double d_expectation_dV_dz(trap_geo params,
                                      double3 pos,
                                      zomplex2 psi) {
    double3 dBdz = dB_dz(pos,
                         params);
    // Create differentiated potential operator
    cuDoubleComplex dVdz[2][2] = {make_cuDoubleComplex(0., 0.)};
    dVdz[0][0] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdz.z,
                                                       0.);
    dVdz[0][1] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdz.x,
                                                       -dBdz.y);
    dVdz[1][0] = 0.5*d_muB*d_gs * make_cuDoubleComplex(dBdz.x,
                                                       dBdz.y);
    dVdz[1][1] = 0.5*d_muB*d_gs * make_cuDoubleComplex(-dBdz.z,
                                                       0.);
    // Perform expectation value calculation
    cuDoubleComplex E_dVdz = cuConj(psi.up) * (psi.up*dVdz[0][0] + psi.dn*dVdz[0][1]) + 
                             cuConj(psi.dn) * (psi.up*dVdz[1][0] + psi.dn*dVdz[1][1]);
    return cuCreal(E_dVdz);
}

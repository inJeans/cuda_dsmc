/** \file
 *  \brief Main code
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef MAGNETIC_FIELD_CUH_INCLUDED
#define MAGNETIC_FIELD_CUH_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "dsmc_utils.cuh"
#include "cuda_dsmc/declare_physical_constants.hpp"

#ifndef MAGNETIC_FIELD_HPP_INCLUDED
#if defined(HARMONIC)
typedef struct FieldParameters {
    double3 omega;
    double B0;
}FieldParams;
#else
typedef struct FieldParameters {
    double B0;
    double max_distribution_width;
}FieldParams;
#endif
#endif  // MAGNETIC_FIELD_HPP_INCLUDED

__device__ double3 dMagneticField(FieldParams params,
                                  double3 pos);

__device__ double3 dMagneticFieldGradient(FieldParams params,
                                          double3 pos);

#endif  // MAGNETIC_FIELD_CUH_INCLUDED

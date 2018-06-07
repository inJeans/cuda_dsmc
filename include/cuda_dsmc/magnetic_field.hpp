/** \file
 *  \brief Main code
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

#ifndef MAGNETIC_FIELD_HPP_INCLUDED
#define MAGNETIC_FIELD_HPP_INCLUDED 1

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "cuda_dsmc/declare_physical_constants.hpp"

extern const double kMaxDistributionWidth;

#if defined(HARMONIC)
typedef struct FieldParameters {
    double3 omega;
    double B0;
}FieldParams;
#else
typedef struct FieldParameters {
    double B0;
}FieldParams;
#endif

double3 magneticField(FieldParams params,
                      double3 pos);

#endif  // MAGNETIC_FIELD_HPP_INCLUDED

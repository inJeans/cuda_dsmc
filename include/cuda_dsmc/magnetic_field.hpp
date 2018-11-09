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

double3 magneticField(FieldParams params,
                      double3 pos);

double3 magneticFieldGradient(FieldParams params,
                              double3 pos);

#endif  // MAGNETIC_FIELD_HPP_INCLUDED

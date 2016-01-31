/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DEFINE_HOST_CONSTANTS_HPP_INCLUDED
#define DEFINE_HOST_CONSTANTS_HPP_INCLUDED 1

#include <cuda_runtime.h>

// PHYSICAL CONSTANTS
double gs   =  0.5;             // Gyromagnetic ratio
double MF   = -1.0;             // Magnetic quantum number
double muB  = 9.27400915e-24;   // Bohr magneton
double mass = 1.443160648e-25;  // 87Rb mass
double pi   = 3.14159265;       // Pi
double a    = 5.3e-9;           // Constant cross-section formula
double kB   = 1.3806503e-23;    // Boltzmann's Constant
double hbar = 1.05457148e-34;   // hbar

// COMPUTATIONAL CONSTANTS
int3 num_cells = make_int3(128, 128, 128);
double3 grid_min = make_double3(0., 0., 0.);
double3 cell_length = make_double3(0., 0., 0.);

#endif  // DEFINE_HOST_CONSTANTS_HPP_INCLUDED

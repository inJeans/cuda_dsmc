/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DECLARE_HOST_CONSTANTS_HPP_INCLUDED
#define DECLARE_HOST_CONSTANTS_HPP_INCLUDED 1

#include <cuda_runtime.h>

// PHYSICAL CONSTANTS
extern double gs;    // Gyromagnetic ratio
extern double MF;    // Magnetic quantum number
extern double muB;   // Bohr magneton
extern double mass;  // 87Rb mass
extern double h_pi;    // Pi
extern double a;     // Constant cross-section formula
extern double kB;    // Boltzmann's Constant
extern double hbar;  // hbar

// COMPUTATIONAL CONSTANTS
extern int3 num_cells;
extern double3 grid_min;
extern double3 cell_length;

#endif  // DECLARE_HOST_CONSTANTS_HPP_INCLUDED

/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DECLARE_DEVICE_CONSTANTS_CUH_INCLUDED
#define DECLARE_DEVICE_CONSTANTS_CUH_INCLUDED 1

// PHYSICAL CONSTANTS
extern __constant__ double d_gs;    // Gyromagnetic ratio
extern __constant__ double d_MF;    // Magnetic quantum number
extern __constant__ double d_muB;   // Bohr magneton
extern __constant__ double d_mass;  // 87Rb mass
extern __constant__ double d_pi;    // Pi
extern __constant__ double d_a;     // Constant cross-section formula
extern __constant__ double d_kB;    // Boltzmann's Constant
extern __constant__ double d_hbar;  // hbar
__constant__ double d_cross_section;

// COMPUTATIONAL CONSTANTS
__device__ double d_FN;
__device__ int3 d_num_cells;
__device__ double3 d_grid_min;
__device__ double3 d_cell_length;
__device__ double3 d_cell_volume;

#endif  // DECLARE_DEVICE_CONSTANTS_CUH_INCLUDED

#include <cuda_runtime.h>

#include "random_numbers.hpp"

__host__ void cu_generate_thermal_velocities(int num_atoms,
                                             double temp,
                                             curandState *state,
                                             double3 *vel);

__global__ void g_generate_thermal_velocities(int num_atoms,
                                              double temp,
                                              curandState *state,
                                              double3 *vel);

__device__ double3 d_thermal_vel(double temp,
                                 curandState *state);

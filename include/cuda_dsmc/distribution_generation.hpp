
#include <cuda_runtime.h>

#include "random_numbers.hpp"

__host__ void generate_thermal_velocities(int num_atoms,
                                          double temp,
                                          curandState *state,
                                          double3 *vel);

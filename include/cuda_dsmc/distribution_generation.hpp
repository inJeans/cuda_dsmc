#ifdef CUDA
#include <cuda_runtime.h>
#endif

#include "random_numbers.hpp"

#ifdef CUDA
__host__ void generate_thermal_velocities(int num_atoms,
                                          double temp,
                                          curandState *state,
                                          double3 *vel);
#endif

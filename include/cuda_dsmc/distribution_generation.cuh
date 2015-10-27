#include <cuda_runtime.h>

 __host__ __device__ double gaussian_point( double mean,
 	                                        double std,
 	                                        double seed );

 __host__ __device__ double uniform_prng( int seed );
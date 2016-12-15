#include "test_helpers.cuh"
#include <stdio.h>

__host__ void uniform_prng_launcher(int num_elements,
                                    curandState *state,
                                    double *h_r) {
    double *d_r;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_r),
                               num_elements*sizeof(double)));
    printf("Num elements = %i\n", num_elements);

    g_uniform_prng<<<1,1>>>(num_elements,
                            state,
                            d_r);

    checkCudaErrors(cudaMemcpy(h_r,
                               d_r,
                               num_elements*sizeof(double),
                               cudaMemcpyDeviceToHost));
    for (int i = 0;
         i < num_elements;
         i += 1) {
        printf("h_r[%i] = %f\n", i, h_r[i]);
    }
    cudaFree(d_r);

    return;
}

__global__ void g_uniform_prng(int num_elements,
                               curandState *state,
                               double *r) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += blockDim.x * gridDim.x) {
        r[i] = curand_uniform(state);
    }

    return;
}

__host__ void gaussian_prng(int num_elements,
                           curandState *state,
                           double *h_r) {
    double *d_r;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_r),
                               num_elements*sizeof(double)));

    g_gaussian_prng<<<num_elements,1>>>(num_elements,
                                        state,
                                        d_r);

    checkCudaErrors(cudaMemcpy(h_r,
                               d_r,
                               num_elements*sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(d_r);

    return;
}

__global__ void g_gaussian_prng(int num_elements,
                                curandState *state,
                                double *r) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += blockDim.x * gridDim.x) {
        r[i] = curand_normal(state);
    }

    return;
}

__host__ void gaussian_point(int num_elements,
                             double mean,
                             double std,
                             curandState *state,
                             double3 *h_p) {
    double3 *d_p;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_p),
                               num_elements*sizeof(double3)));

    g_gaussian_point<<<num_elements,1>>>(num_elements,
                                         mean,
                                         std,
                                         state,
                                         d_p);

    checkCudaErrors(cudaMemcpy(h_p,
                               d_p,
                               num_elements*sizeof(double3),
                               cudaMemcpyDeviceToHost));

    return;
}

/** \fn __global__ void g_gaussian_point(int num_elements,
 *                                       double mean,
 *                                       double std,
 *                                       curandState *state,
 *                                       double3 *p) 
 *  \brief Fills a `double3` array where each element is a three dimensional
 *  normally distributed point in cartesian space, with 
 *  mean and std as the mean and standard deviation respectively.
 *  \param mean Gaussian mean of the components of the output double3.
 *  \param std Standard deviation of the components of the output double3.
 *  \param *state Array of state vectors for the rng.
 *  \param *p Output array of normally distributed points.
 *  \exception not yet.
 *  \return A gaussian distributed `double3` point in cartesian space.
*/

__global__ void g_gaussian_point(int num_elements,
                                    double mean,
                                    double std,
                                    curandState *state,
                                    double3 *p) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += blockDim.x * gridDim.x) {
        p[i] = d_gaussian_point(mean,
                                std,
                                &state[i]);
    }
}

__global__ void zero_elements(int num_elements,
                               double *array) {
    for (int element = blockIdx.x * blockDim.x + threadIdx.x;
         element < num_elements;
         element += blockDim.x * gridDim.x)
        array[element] = 0.;

    return;
}

__global__ void negative_elements(int num_elements,
                                  int2 *array) {
    for (int element = blockIdx.x * blockDim.x + threadIdx.x;
         element < num_elements;
         element += blockDim.x * gridDim.x)
        array[element] = make_int2(-1, -1);

    return;
}

__host__ void cu_nan_checker(int num_atoms,
                             double3 *array) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the nan "
                "checker kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_nan_checker,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif

    g_nan_checker<<<grid_size,
                       block_size>>>
                      (num_atoms,
                       array);

    return;
}

__global__ void g_nan_checker(int num_atoms,
                              double3 *array) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        if(array[atom].x != array[atom].x) {
            printf("Nan - array[%i] = {%g, %g, %g}\n",
                    atom, array[atom].x, array[atom].y, array[atom].z);
        } else if (array[atom].y != array[atom].y) {
            printf("Nan - array[%i] = {%g, %g, %g}\n",
                    atom, array[atom].x, array[atom].y, array[atom].z);
        } else if (array[atom].z != array[atom].z) {
            printf("Nan - array[%i] = {%g, %g, %g}\n",
                    atom, array[atom].x, array[atom].y, array[atom].z);
        }
    }

    return;
}

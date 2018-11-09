/** \file
 *  \brief Utility functions
 *
 *  All the common utility functions I need for doing stuff like copying,
 *  collating results from multiple devices/streams/threads, saving arrays
 *  Things of that nature
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "cuda_dsmc/dsmc_utils.cuh"

/** \brief Evenly divide elements amongst parallel unit
 *
 *  \param num_arrays Number of ranks in the MPI world.
 *  \param num_elements Pointer to the global number of elements.
 *  \exception not yet.
 *  \return The rank local number of elements
 */
void combineDeviceArrays(int num_devices,
                         int num_elements,
                         double3** device_arrays,
                         double3* host_array) {
    
    int element_sum = 0;
    for (int d=0; d < num_devices; ++d) {
        int num_local_elements = num_elements;
        numberElementsPerParallelUnit(d,
                                      num_devices,
                                      &num_local_elements);
        CUDA_CALL(cudaMemcpy(host_array+element_sum,
                             device_arrays[d],
                             num_local_elements * sizeof(double3),
                             cudaMemcpyDeviceToHost)); 
    }

    return;
}

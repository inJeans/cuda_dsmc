/** \file
 *  \brief Unit tests for the collisions file
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "ehrenfest_test.cuh"

double fractional_tol = 0.05; 

SCENARIO("[DEVICE] Execute a full ehrenfest simulation", "[d-ehrenfest]") {
    GIVEN("A thermally distributed intial distribution.") {

        // Initialise trapping parameters
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the trapping parameters.");
#endif
#if defined(IP)  // Ioffe Pritchard trap
        trap_geo trap_parameters;
        trap_parameters.B0 = 0.01;
        trap_parameters.dB = 20.;
        trap_parameters.ddB = 40000.;
#else  // Quadrupole trap
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;
#endif

        // Initialise computational parameters
        int num_atoms = 1e5;
        FN = 10;
        double dt = 1.e-6;
        int num_time_steps = 10;
        double init_temp = 20.e-6;

        // Initialise grid parameters
        k_num_cells = make_int3(35, 35, 35);
        total_num_cells = k_num_cells.x*k_num_cells.y*k_num_cells.z;

    // Initialise rng
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the rng state array.");
        LOGF(DEBUG, "\nAllocating %i curandState elements on the device.",
             num_atoms);
#endif
        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_atoms*sizeof(curandState)));
        initialise_rng_states(num_atoms,
                              state,
                              false);

        // Initialise atom_id
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the atom_id array.");
        LOGF(DEBUG, "\nAllocating %i int elements on the device.",
             num_atoms);
#endif
        int *atom_id;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&atom_id),
                                   num_atoms*sizeof(int)));
        initialise_atom_id(num_atoms,
                           atom_id);

        // Initialise cell_id
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the cell_id array.");
        LOGF(DEBUG, "\nAllocating %i int elements on the device.",
             num_atoms);
#endif
        int *cell_id;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_id),
                                   num_atoms*sizeof(int)));
        checkCudaErrors(cudaMemset(cell_id,
                                   0,
                                   num_atoms*sizeof(int)));

        // Initialise cell_start_end
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the cell_start_end array.");
        LOGF(DEBUG, "\nAllocating %i int2 elements on the device.",
             total_num_cells+1);
#endif
        int2 *cell_start_end;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_start_end),
                                   (total_num_cells+1)*sizeof(int2)));
        checkCudaErrors(cudaMemset(cell_start_end,
                                   -1,
                                   (total_num_cells+1)*sizeof(int2)));

        // Initialise cell_num_atoms
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the cell_num_atoms array.");
        LOGF(DEBUG, "\nAllocating %i int elements on the device.",
             total_num_cells+1);
#endif
        int *cell_num_atoms;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_num_atoms),
                                   (total_num_cells+1)*sizeof(int)));
        checkCudaErrors(cudaMemset(cell_num_atoms,
                                   0,
                                   (total_num_cells+1)*sizeof(int)));

        // Initialise cell_cumulative_num_atoms
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the cell_cumulative_num_atoms array.");
        LOGF(DEBUG, "\nAllocating %i int elements on the device.",
             total_num_cells+1);
#endif
        int *cell_cumulative_num_atoms;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cell_cumulative_num_atoms),
                                   (total_num_cells+1)*sizeof(int)));
        checkCudaErrors(cudaMemset(cell_cumulative_num_atoms,
                                   0,
                                   (total_num_cells+1)*sizeof(int)));

        // Initialise collision_count
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the collision_count array.");
        LOGF(DEBUG, "\nAllocating %i int elements on the device.",
             total_num_cells);
#endif
        int *collision_count;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&collision_count),
                                   (total_num_cells)*sizeof(int)));
        checkCudaErrors(cudaMemset(collision_count,
                                   0,
                                   (total_num_cells)*sizeof(int)));

        // Initialise collision_remainder
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the collision_remainder array.");
        LOGF(DEBUG, "\nAllocating %i int elements on the device.",
             total_num_cells);
#endif
        double *collision_remainder;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&collision_remainder),
                                   (total_num_cells)*sizeof(double)));
        zero_elements<<<total_num_cells,1>>>(total_num_cells,
                                             collision_remainder);

        // Initialise sig_vr_max
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the sig_vr_max array.");
        LOGF(DEBUG, "\nAllocating %i int elements on the device.",
             total_num_cells);
#endif
        double *sig_vr_max;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&sig_vr_max),
                                   (total_num_cells)*sizeof(double)));
        zero_elements<<<total_num_cells,1>>>(total_num_cells,
                                             sig_vr_max);

    // Initialise velocities
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the velocity array.");
        LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
             num_atoms);
#endif
        double3 *vel;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&vel),
                                   num_atoms*sizeof(double3)));

        // Generate velocity distribution
        generate_thermal_velocities(num_atoms,
                                    init_temp,
                                    state,
                                    vel);

        // Initialise positions
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the position array.");
        LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
             num_atoms);
#endif
        double3 *pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&pos),
                                   num_atoms*sizeof(double3)));

        // Generate position distribution
        generate_thermal_positions(num_atoms,
                                   init_temp,
                                   trap_parameters,
                                   state,
                                   pos);

        // Initialise wavefunction
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the wavefunction array.");
        LOGF(DEBUG, "\nAllocating %i wavefunction elements on the device.",
             num_atoms);
#endif  // Logging
        wavefunction *psi;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&psi),
                                   num_atoms*sizeof(wavefunction)));

        // Generate wavefunction
        generate_aligned_spins(num_atoms,
                               trap_parameters,
                               pos,
                               psi);

        // Initialise accelerations
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the acceleration array.");
        LOGF(DEBUG, "\nAllocating %i double3 elements on the device.",
             num_atoms);
#endif
        double3 *acc;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&acc),
                                   num_atoms*sizeof(double3)));

        // Generate accelerations
        update_accelerations(num_atoms,
                             trap_parameters,
                             pos,
                             acc,
                             psi);

        // Initialise kinetic energy
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the kinetic energy array.");
        LOGF(DEBUG, "\nAllocating %i double elements on the device.",
             num_atoms);
#endif
        double *d_kinetic_energy;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_kinetic_energy),
                                   num_atoms*sizeof(double)));
        double *avg_kinetic_energy;
        avg_kinetic_energy = reinterpret_cast<double*>(calloc(num_atoms,
                                                       sizeof(double)));

#if defined(LOGGING)
        LOGF(DEBUG, "\nCreating the cuBLAS handle.\n");
#endif
        cublasHandle_t cublas_handle;
        checkCudaErrors(cublasCreate(&cublas_handle));

        // Set up global grid parameters
        initialise_grid_params(num_atoms,
                               cublas_handle,
                               pos);

        // Evolve many time step
#if defined(LOGGING)
        LOGF(INFO, "\nEvolving distribution for %i time steps.", num_time_steps);
#endif
	for (int t = 0; t < num_time_steps; ++t) {
	    velocity_verlet_update(num_atoms,
                                   dt,
                                   trap_parameters,
                                   cublas_handle,
                                   pos,
                                   vel,
                                   acc,
                                   psi);
            collide_atoms(num_atoms,
                          total_num_cells,
                          dt,
                          pos,
                          vel,
                          state,
                          sig_vr_max,
                          cell_id,
                          atom_id,
                          cell_start_end,
                          cell_num_atoms,
                          cell_cumulative_num_atoms,
                          collision_remainder,
                          collision_count);
            avg_kinetic_energy[t] = inst_kinetic_energy(num_atoms,
                                                        vel,
                                                        d_kinetic_energy) /
                                    num_atoms;

            progress_bar(t,
                         num_time_steps);
        }
        printf("\n");
        for (int i = 0; i < num_time_steps; ++i)
            printf("avg_kinetic_energy[%i] = %g uK\n", i,
                               avg_kinetic_energy[i]/kB*1.e6);

#if defined(LOGGING)
        LOGF(DEBUG, "\nDestroying the cuBLAS handle.\n");
#endif
        checkCudaErrors(cublasDestroy(cublas_handle));

#if defined(LOGGING)
        LOGF(INFO, "\nCleaning up device memory.");
#endif
        cudaFree(state);
        cudaFree(atom_id);
        cudaFree(cell_id);
        cudaFree(cell_start_end);
        cudaFree(cell_num_atoms);
        cudaFree(cell_cumulative_num_atoms);
        cudaFree(collision_count);
        cudaFree(collision_remainder);
        cudaFree(sig_vr_max);
        cudaFree(vel);
        cudaFree(pos);
        cudaFree(acc);
        cudaFree(psi);
        cudaFree(d_kinetic_energy);

        free(avg_kinetic_energy);
    }
}

__host__ double inst_kinetic_energy(int num_atoms,
                                    double3 *vel,
                                    double *kinetic_energy) {
    int NUM_BATCHES = 10;
    double *h_inst_kin = NULL;
    h_inst_kin = reinterpret_cast<double*>(calloc(1,
                                                  sizeof(double)));
    double *b_inst_kin = NULL;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&b_inst_kin),
                               NUM_BATCHES*sizeof(double)));
    double *d_inst_kin = NULL;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_inst_kin),
                               sizeof(double)));
    int *h_offsets = NULL;
    h_offsets = reinterpret_cast<int*>(calloc(NUM_BATCHES+1,
                                                 sizeof(int)));
    int *d_offsets = NULL;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_offsets),
                               (NUM_BATCHES+1)*sizeof(int)));
    for (int b = 0; b < NUM_BATCHES; ++b) {
        h_offsets[b] = b * num_atoms / NUM_BATCHES;
    }
    h_offsets[NUM_BATCHES+1] = num_atoms;
    checkCudaErrors(cudaMemcpy(d_offsets,
                               h_offsets,
                               (NUM_BATCHES+1)*sizeof(int),
                               cudaMemcpyHostToDevice));

    cu_kinetic_energy(num_atoms,
                      vel,
                      kinetic_energy);
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    checkCudaErrors(cub::DeviceSegmentedReduce::Sum(d_temp_storage,
                                                    temp_storage_bytes,
                                                    kinetic_energy,
                                                    b_inst_kin,
                                                    NUM_BATCHES,
                                                    d_offsets,
                                                    d_offsets+1));
    // Allocate temporary storage
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp_storage),
                               temp_storage_bytes));
    // Run sum-reduction
    checkCudaErrors(cub::DeviceSegmentedReduce::Sum(d_temp_storage,
                                                    temp_storage_bytes,
                                                    kinetic_energy,
                                                    b_inst_kin,
                                                    NUM_BATCHES,
                                                    d_offsets,
                                                    d_offsets+1));
    // Reduce batches
    // Run sum-reduction
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage,
                                           temp_storage_bytes,
                                           b_inst_kin,
                                           d_inst_kin,
                                           NUM_BATCHES));
    checkCudaErrors(cudaMemcpy(h_inst_kin,
                               d_inst_kin,
                               1.*sizeof(double),
                               cudaMemcpyDeviceToHost));
    cudaFree(d_temp_storage);
    cudaFree(d_inst_kin);
    cudaFree(b_inst_kin);
    cudaFree(d_offsets);
    free(h_offsets);
    return h_inst_kin[0];
}

__host__ void cu_kinetic_energy(int num_atoms,
                                double3 *vel,
                                double *kinetic_energy) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the kinetic "
                "energy calculation kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_kinetic_energy,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif

    g_kinetic_energy<<<grid_size,
                       block_size>>>
                      (num_atoms,
                       vel,
                       kinetic_energy);

    return;
}

__global__ void g_kinetic_energy(int num_atoms,
                                 double3 *vel,
                                 double *kinetic_energy) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        kinetic_energy[atom] = d_kinetic_energy(vel[atom]);
    }

    return;
}

__device__ double d_kinetic_energy(double3 vel) {
    return 0.5 * d_mass * norm(vel) * norm(vel);
}

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
        double dt = 1.e-7;
        int num_time_steps = 2;
        double init_temp = 20.e-6;

        // Initialise grid parameters
        k_num_cells = make_int3(35, 35, 35);
        total_num_cells = k_num_cells.x*k_num_cells.y*k_num_cells.z;

        int largest = 0;
        if (num_atoms > total_num_cells) {
            largest = num_atoms;
        } else {
            largest = total_num_cells;
        }

    // Initialise rng
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the rng state array.");
        LOGF(DEBUG, "\nAllocating %i curandState elements on the device.",
             largest);
#endif
        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   largest*sizeof(curandState)));
        initialise_rng_states(largest,
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
        avg_kinetic_energy = reinterpret_cast<double*>(calloc(num_time_steps+1,
                                                              sizeof(double)));

        // Initialise potential energy
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the potential energy array.");
        LOGF(DEBUG, "\nAllocating %i double elements on the device.",
             num_atoms);
#endif
        double *d_potential_energy;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_potential_energy),
                                   num_atoms*sizeof(double)));
        double *avg_potential_energy;
        avg_potential_energy = reinterpret_cast<double*>(calloc(num_time_steps+1,
                                                                sizeof(double)));

        // Initialise projection
#if defined(LOGGING)
        LOGF(INFO, "\nInitialising the projection array.");
        LOGF(DEBUG, "\nAllocating %i double elements on the device.",
             num_atoms);
#endif
        double *d_projection;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_projection),
                                   num_atoms*sizeof(double)));
        double *avg_projection;
        avg_projection = reinterpret_cast<double*>(calloc(num_time_steps+1,
                                                          sizeof(double)));

        // initialise time
        double *sim_time;
        sim_time = reinterpret_cast<double*>(calloc(num_time_steps+1,
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

        double3 *h_pos;
        h_pos = reinterpret_cast<double3*>(calloc(num_atoms,
                                                  sizeof(double3)));
        checkCudaErrors(cudaMemcpy(h_pos,
                                   pos,
                                   num_atoms*sizeof(double3),
                                   cudaMemcpyDeviceToHost));
        FILE *init_pos_file_pointer = fopen("initial_position.data", "w");
        for (int i=0; i<num_time_steps+1; ++i) {
            fprintf(init_pos_file_pointer, "%g, %g, %g\n", h_pos[i].x,
                                                           h_pos[i].y,
                                                           h_pos[i].z);
        }
        fclose(init_pos_file_pointer);

        int *h_collision_count;
        h_collision_count = reinterpret_cast<int*>(calloc(total_num_cells,
                                                          sizeof(int)));

        FILE *collision_file_pointer = fopen("collision.data", "w");
        for (int i=0; i<total_num_cells+1; ++i) {
            fprintf(collision_file_pointer, "%i\t", h_collision_count[i]);
        }
        fprintf(collision_file_pointer, "\n");
        fclose(collision_file_pointer);

        sim_time[0] = 0.;

        avg_kinetic_energy[0] = inst_kinetic_energy(num_atoms,
                                                    vel,
                                                    d_kinetic_energy) /
                                 num_atoms;
        avg_potential_energy[0] = inst_potential_energy(num_atoms,
                                                        pos,
                                                        trap_parameters,
                                                        psi,
                                                        d_potential_energy) /
                                  num_atoms;

        avg_projection[0] = inst_projection(num_atoms,
                                            pos,
                                            trap_parameters,
                                            psi,
                                            d_projection) / num_atoms;

        // Evolve many time step
#if defined(LOGGING)
        LOGF(INFO, "\nEvolving distribution for %i time steps.", num_time_steps);
#endif
        for (int t = 0; t < num_time_steps; ++t) {
            for(int u=0; u<1000; ++u) {
                velocity_verlet_update(num_atoms,
                                       dt,
                                       trap_parameters,
                                       cublas_handle,
                                       pos,
                                       vel,
                                       acc,
                                       psi);
            }
            sim_time[t+1] = sim_time[t] + 1000*dt;
            collide_atoms(num_atoms,
                          total_num_cells,
                          1000*dt,
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

            checkCudaErrors(cudaMemcpy(h_collision_count,
                                       collision_count,
                                       total_num_cells*sizeof(int),
                                       cudaMemcpyDeviceToHost));
            collision_file_pointer = fopen("collision.data", "a");
            for (int i=0; i<total_num_cells+1; ++i) {
                fprintf(collision_file_pointer, "%i\t", h_collision_count[i]);
            }
            fprintf(collision_file_pointer, "\n");
            fclose(collision_file_pointer);

            avg_kinetic_energy[t+1] = inst_kinetic_energy(num_atoms,
                                                          vel,
                                                          d_kinetic_energy) /
                                      num_atoms;
            avg_potential_energy[t+1] = inst_potential_energy(num_atoms,
                                                              pos,
                                                              trap_parameters,
                                                              psi,
                                                              d_potential_energy) /
                                         num_atoms;

            avg_projection[t+1] = inst_projection(num_atoms,
                                                  pos,
                                                  trap_parameters,
                                                  psi,
                                                  d_projection) / num_atoms;

            progress_bar(t,
                         num_time_steps);
        }
        printf("\n");
        for (int i = 0; i < num_time_steps+1; ++i) {
            printf("avg_kinetic_energy[%i] = %g uK\n", i,
                               avg_kinetic_energy[i]/kB*1.e6);
            printf("avg_potential_energy[%i] = %g uK\n", i,
                               avg_potential_energy[i]/kB*1.e6);
            printf("avg_projection[%i] = %g\n", i, avg_projection[i]);
        }

        FILE *time_file_pointer = fopen("time.data", "w");
        FILE *kinetic_file_pointer = fopen("kinetic_energy.data", "w");
        FILE *potential_file_pointer = fopen("potential_energy.data", "w");
        FILE *projection_file_pointer = fopen("projection.data", "w");
        for (int i=0; i<num_time_steps+1; ++i) {
            fprintf(time_file_pointer, "%g\n", sim_time[i]);
            fprintf(kinetic_file_pointer, "%g\n", avg_kinetic_energy[i]/kB*1.e6);
            fprintf(potential_file_pointer, "%g\n", avg_potential_energy[i]/kB*1.e6);
            fprintf(projection_file_pointer, "%g\n", avg_projection[i]);
        }
        fclose(time_file_pointer);
        fclose(kinetic_file_pointer);
        fclose(potential_file_pointer);
        fclose(projection_file_pointer);

        FILE *final_pos_file_pointer = fopen("final_position.data", "w");
        for (int i=0; i<num_time_steps+1; ++i) {
            fprintf(final_pos_file_pointer, "%g, %g, %g\n", h_pos[i].x,
                                                            h_pos[i].y,
                                                            h_pos[i].z);
        }
        fclose(final_pos_file_pointer);

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
        cudaFree(d_potential_energy);
        cudaFree(d_projection);

        free(avg_kinetic_energy);
        free(avg_potential_energy);
        free(avg_projection);
        free(sim_time);
        free(h_pos);
    }
}

__host__ double inst_kinetic_energy(int num_atoms,
                                    double3 *vel,
                                    double *kinetic_energy) {
    double *h_inst_kin = NULL;
    h_inst_kin = reinterpret_cast<double*>(calloc(1,
                                                  sizeof(double)));
    double *d_inst_kin = NULL;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_inst_kin),
                               sizeof(double)));

    cu_kinetic_energy(num_atoms,
                      vel,
                      kinetic_energy);
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage,
                                           temp_storage_bytes,
                                           kinetic_energy,
                                           d_inst_kin,
                                           num_atoms));
    // Allocate temporary storage
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp_storage),
                               temp_storage_bytes));
    // Run sum-reduction
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage,
                                           temp_storage_bytes,
                                           kinetic_energy,
                                           d_inst_kin,
                                           num_atoms));
    checkCudaErrors(cudaMemcpy(h_inst_kin,
                               d_inst_kin,
                               1.*sizeof(double),
                               cudaMemcpyDeviceToHost));
    cudaFree(d_temp_storage);
    cudaFree(d_inst_kin);

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
        if(kinetic_energy[atom] != kinetic_energy[atom]) {
            kinetic_energy[atom] = 0.;
            vel[atom] = make_double3(0., 0., 0.);
        }
    }

    return;
}

__device__ double d_kinetic_energy(double3 vel) {
    return 0.5 * d_mass * norm(vel) * norm(vel);
}

__host__ double inst_potential_energy(int num_atoms,
                                      double3 *pos,
                                      trap_geo params,
                                      wavefunction *psi,
                                      double *potential_energy) {
    double *h_inst_pot = NULL;
    h_inst_pot = reinterpret_cast<double*>(calloc(1,
                                                  sizeof(double)));
    double *d_inst_pot = NULL;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_inst_pot),
                               sizeof(double)));

    cu_potential_energy(num_atoms,
                       pos,
                       params,
                       psi,
                       potential_energy);
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage,
                                           temp_storage_bytes,
                                           potential_energy,
                                           d_inst_pot,
                                           num_atoms));
    // Allocate temporary storage
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp_storage),
                               temp_storage_bytes));
    // Run sum-reduction
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage,
                                           temp_storage_bytes,
                                           potential_energy,
                                           d_inst_pot,
                                           num_atoms));
    checkCudaErrors(cudaMemcpy(h_inst_pot,
                               d_inst_pot,
                               1.*sizeof(double),
                               cudaMemcpyDeviceToHost));
    cudaFree(d_temp_storage);
    cudaFree(d_inst_pot);

    return h_inst_pot[0];
}

__host__ void cu_potential_energy(int num_atoms,
                                  double3 *pos,
                                  trap_geo params,
                                  wavefunction *psi,
                                  double *potential_energy) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the potential "
                "energy calculation kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_potential_energy,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif

    g_potential_energy<<<grid_size,
                         block_size>>>
                        (num_atoms,
                         pos,
                         params,
                         psi,
                         potential_energy);

    return;
}

__global__ void g_potential_energy(int num_atoms,
                                   double3 *pos,
                                   trap_geo params,
                                   wavefunction *psi,
                                   double *potential_energy) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        potential_energy[atom] = d_potential_energy(pos[atom],
                                                    params,
                                                    psi[atom]);
        if(potential_energy[atom] != potential_energy[atom]) {
            potential_energy[atom] = 0.;
            pos[atom] = make_double3(0., 0., 0.);
            psi[atom] = make_wavefunction(0., 0., 0., 0., true);
        }
    }

    return;
}

__device__ double d_potential_energy(double3 pos,
                                     trap_geo params,
                                     wavefunction psi) {
    double3 local_B = B(pos,
                        params);
    cuDoubleComplex H[2][2] = {make_cuDoubleComplex(0., 0.)};
    H[0][0] = 0.5*d_gs*d_muB * make_cuDoubleComplex(local_B.z,
                                                    0.);
    H[0][1] = 0.5*d_gs*d_muB * make_cuDoubleComplex(local_B.x,
                                                    -local_B.y);
    H[1][0] = 0.5*d_gs*d_muB * make_cuDoubleComplex(local_B.x,
                                                    local_B.y);
    H[1][1] = 0.5*d_gs*d_muB * make_cuDoubleComplex(-local_B.z,
                                                    0.);
    cuDoubleComplex potential = psi.up*(H[0][0]*cuConj(psi.up) + H[1][0]*cuConj(psi.dn)) +
                                psi.dn*(H[0][1]*cuConj(psi.up) + H[1][1]*cuConj(psi.dn));

    return cuCreal(potential);
}

__host__ double inst_projection(int num_atoms,
                                double3 *pos,
                                trap_geo params,
                                wavefunction *psi,
                                double *projection) {
    double *h_inst_proj = NULL;
    h_inst_proj = reinterpret_cast<double*>(calloc(1,
                                                  sizeof(double)));
    double *d_inst_proj = NULL;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_inst_proj),
                               sizeof(double)));

    cu_projection(num_atoms,
                  pos,
                  params,
                  psi,
                  projection);
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage,
                                           temp_storage_bytes,
                                           projection,
                                           d_inst_proj,
                                           num_atoms));
    // Allocate temporary storage
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp_storage),
                               temp_storage_bytes));
    // Run sum-reduction
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage,
                                           temp_storage_bytes,
                                           projection,
                                           d_inst_proj,
                                           num_atoms));
    checkCudaErrors(cudaMemcpy(h_inst_proj,
                               d_inst_proj,
                               1.*sizeof(double),
                               cudaMemcpyDeviceToHost));
    cudaFree(d_temp_storage);
    cudaFree(d_inst_proj);

    return h_inst_proj[0];
}

__host__ void cu_projection(int num_atoms,
                            double3 *pos,
                            trap_geo params,
                            wavefunction *psi,
                            double *projection) {
#if defined(LOGGING)
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the projection "
                "calculation kernel.\n");
#endif
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_projection,
                                       0,
                                       num_atoms);
    grid_size = (num_atoms + block_size - 1) / block_size;
#if defined(LOGGING)
    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
#endif

    g_projection<<<grid_size,
                   block_size>>>
                  (num_atoms,
                   pos,
                   params,
                   psi,
                   projection);

    return;
}

__global__ void g_projection(int num_atoms,
                             double3 *pos,
                             trap_geo params,
                             wavefunction *psi,
                             double *projection) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        projection[atom] = d_projection(pos[atom],
                                        params,
                                        psi[atom]);
        if(projection[atom] != projection[atom]) {
            projection[atom] = 0.;
            pos[atom] = make_double3(0., 0., 0.);
            psi[atom] = make_wavefunction(0., 0., 0., 0., true);
        }
    }

    return;
}

__device__ double d_projection(double3 pos,
                               trap_geo params,
                               wavefunction psi) {
    double3 Bn = unit(B(pos,
                        params));
    cuDoubleComplex P = make_cuDoubleComplex(0., 0.);
    P = 0.5 * (((1.-Bn.z)*psi.dn + Bn.x*psi.up)*cuConj(psi.dn) +
               ((1.+Bn.z)*psi.up + Bn.x*psi.dn)*cuConj(psi.up)) -
        Bn.y*cuCimag(psi.up*cuConj(psi.dn));;

    return cuCreal(P);
}

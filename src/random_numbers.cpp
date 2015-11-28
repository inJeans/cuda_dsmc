/** \file
 *  \brief Functions necessary for generating a thermal distribution
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "random_numbers.hpp"
#include "random_numbers.cuh"

/** \fn void initialise_rng_states(int n_states,
                                   curandState *state) 
 *  \brief Calls the function to fill an array of rng states.
 *  \param n_states Number of rng states.
 *  \param *state Pointer to array of length n_states.
 *  \exception not yet.
 *  \return void
*/

void initialise_rng_states(int n_states,
                           curandState *state) {
#ifdef CUDA
    cu_initialise_rng_states(n_states,
                             state);
#endif

    return;
}

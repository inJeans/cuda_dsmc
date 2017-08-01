/** \file
 *  \brief Random number generators and associated random functions
 *
 *  Here we define all the functions required for random number generation,
 *  from the generation of seeds and initialisation of rngs to the generation
 *  of random vectors in cartesian space.
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "cuda_dsmc/random_numbers.hpp"

/** \fn double uniform_random(pcg32x2_random_t* rng)
 *  \brief Generates a uniformly distributed random number between [0, 1)
 *
 *  This function makes a call to generate a 64-bit random integer and then
 *  converts this integer into a double.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return A random number in the range [0, 1)
 */
double uniform_random(pcg32x2_random_t* rng) {
    uint64_t r = pcg32x2_random_r(rng);

    return ldexp(r, -64);
}

/** \fn void pcg32x2_srandom_r(pcg32x2_random_t* rng,
 *                             uint64_t seed1,
 *                             uint64_t seed2,
 *                             uint64_t seq1,
 *                             uint64_t seq2)
 *  \brief Initialise the random number generator using the supplied seeds
 *
 *  Since we are using our custom random number generator type that combines
 *  two distinct rng streams, we require to independant seeds so that we can
 *  initialise each independant stream.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \param seed1 A unique seed for the first of the rng streams.
 *  \param seed2 A unique seed for the second of the rng streams.
 *  \param seq1 Output sequence for the first of the rng streams.
 *  \param seq2 Output sequence for the second of the rng streams.
 *  \exception not yet.
 */
void pcg32x2_srandom_r(pcg32x2_random_t* rng,
                       uint64_t seed1,
                       uint64_t seed2,
                       uint64_t seq1,
                       uint64_t seq2) {
    uint64_t mask = ~0ull >> 1;
    // The stream for each of the two generators *must* be distinct
    if ((seq1 & mask) == (seq2 & mask))
        seq2 = ~seq2;

    pcg32_srandom_r(rng->gen,
                    seed1,
                    seq1);
    pcg32_srandom_r(rng->gen+1,
                    seed2,
                    seq2);

    return;
}

/** \fn uint64_t pcg32x2_random_r(pcg32x2_random_t* rng)
 *  \brief Generates a random 64-bit integer.
 *
 *  To create a 64-bit random integer we can combine two 32-bit random integers.
 *  \param rng A pointer to our custom random number generator type that contains
 *  two distinct rng streams.
 *  \exception not yet.
 *  \return A random 64-bit integer
 */
uint64_t pcg32x2_random_r(pcg32x2_random_t* rng) {
    return ((uint64_t)(pcg32_random_r(rng->gen)) << 32)
           | pcg32_random_r(rng->gen+1);
}

/** \file
 *  \brief file description
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

#include "random_numbers.hpp"

int main(int argc, char const *argv[]) {
    printf("****************************\n");
    printf("*                          *\n");
    printf("*   WELCOME TO CUDA DSMC   *\n");
    printf("*                          *\n");
    printf("****************************\n");

    // Generate distribution.
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, 42u, 54u);
    double3 p = gaussian_point(0., 1., &rng);
    printf("p = { %f,%f,%f }\n", p.x, p.y, p.z);

    return 0;
}

/** \fn void doxygen_test( double x )
 *  \brief Short description
 *  \param x double that gets printed
 *  \warning What does this do?
 *  Detailed description starts here.
 *  \return void
 */

void doxygen_test(double x) {
    printf("%f\n", x);
    return;
}

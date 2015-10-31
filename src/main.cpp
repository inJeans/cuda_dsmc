/** \file
 *  \brief file description
 *
 *  More detailed description
 */

#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

#include "distribution_generation.cuh"

int main(int argc, char const *argv[])
{
	printf( "****************************\n" );
	printf( "*                          *\n" );
	printf( "*   WELCOME TO CUDA DSMC   *\n" );
	printf( "*                          *\n" );
	printf( "****************************\n" );

	FILE *fp;
	fp = fopen("test_numbers.txt", "a+");

	// Generate distribution.
	pcg32_random_t rng;
    pcg32_srandom_r(&rng, 42u, 54u);

    for (int i = 0; i < 1000; ++i)
    {
    	fprintf( fp, "%f, ", gaussian_point( 0., 1., &rng ) );
    }

	fclose(fp);

	return 0;
}

/** \fn void doxygen_test( double x )
 *  \brief Short description
 *  \param x double that gets printed
 *  \warning What does this do?
 *  Detailed description starts here.
 *  \return void
 */

void doxygen_test( double x )
{
	printf("%f\n", x);
	return;
}
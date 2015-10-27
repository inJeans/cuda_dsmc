/** \file
 *  \brief file description
 *
 *  More detailed description
 */

#include <stdio.h>
#include <cuda_runtime.h>

#include "distribution_generation.cuh"

int main(int argc, char const *argv[])
{
	printf( "****************************\n" );
	printf( "*                          *\n" );
	printf( "*   WELCOME TO CUDA DSMC   *\n" );
	printf( "*                          *\n" );
	printf( "****************************\n" );

	// Generate distribution.

	printf( "\n%f\n", gaussian_point( 0., 1., 0. ) );

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
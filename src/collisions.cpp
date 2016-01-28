/** \file
 *  \brief Functions necessary for colliding a distribution of atoms
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include "collisions.hpp"
#ifdef CUDA
#include "collisions.cuh"
#endif

#include "declare_host_constants.hpp"
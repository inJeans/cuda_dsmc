/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef EHRENFEST_TEST_HPP_INCLUDED
#define EHRENFEST_TEST_HPP_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"

#if defined(LOGGING)
#include <g3log/g3log.hpp>
#endif

#include "catch.hpp"

#include "distribution_generation.hpp"
#include "collisions.hpp" 
#include "distribution_evolution.hpp"

#include "utilities.hpp"
#include "test_helpers.cuh"
#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

#endif  // EHRENFEST_TEST_HPP_INCLUDED
/**
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#ifndef DISTRIBUTION_EVOLUTION_TESTS_CUH_INCLUDED
#define DISTRIBUTION_EVOLUTION_TESTS_CUH_INCLUDED 1

double mean_x(double3 *array,
              int num_elements);

double mean_y(double3 *array,
              int num_elements);

double mean_z(double3 *array,
              int num_elements);

double std_dev_x(double3 *array,
                 int num_elements);

double std_dev_y(double3 *array,
                 int num_elements);

double std_dev_z(double3 *array,
                 int num_elements);

#endif  // DISTRIBUTION_EVOLUTION_TESTS_CUH_INCLUDED

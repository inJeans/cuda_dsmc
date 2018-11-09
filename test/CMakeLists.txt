cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project (cuda_dsmc_test)

# User defined GoogleTest library destintation
IF(GOOGLETEST_LIBRARY_PATH)
    link_directories("${GOOGLETEST_LIBRARY_PATH}")
ENDIF()

# User defined GoogleTest header destintation
IF(GOOGLETEST_INCLUDE_PATH)
    include_directories("${GOOGLETEST_INCLUDE_PATH}")
ENDIF()

# User defined testu01 library destintation
IF(TESTU01_LIBRARY_PATH)
    link_directories("${TESTU01_LIBRARY_PATH}")
ENDIF()

# User defined testu01 header destintation
IF(TESTU01_INCLUDE_PATH)
    include_directories("${TESTU01_INCLUDE_PATH}")
ENDIF()

# User defined testu01 library destintation
IF(PCG_LIBRARY_PATH)
    link_directories("${PCG_LIBRARY_PATH}")
ENDIF()

# User defined testu01 header destintation
IF(PCG_INCLUDE_PATH)
    include_directories("${PCG_INCLUDE_PATH}")
ENDIF()

# MAIN TESTS
IF(CUDA)
    SET(SOURCES main_execution_test.cpp )
    cuda_add_executable(main_test ${SOURCES})
    target_link_libraries(main_test gtest gtest_main g3logger -lpthread -lm)
    target_compile_definitions(main_test PRIVATE -DCUDA)
ELSE()
    SET(SOURCES main_execution_test.cpp)
    cuda_add_executable(main_test ${SOURCES})
    target_link_libraries(main_test gtest gtest_main g3logger -lpthread -lm)
ENDIF()

add_test(main_execution main_test)

# RNG TESTS
IF(CUDA)
    MESSAGE(WARNING "Compiling device rng tests")
    SET(SOURCES random_numbers_test.cu
                ../src/dsmc_utils.cpp
                ../src/magnetic_field.cpp
                ../src/random_numbers.cpp
                ../src/random_numbers.cu)
    cuda_add_executable(rng_test ${SOURCES})
    target_link_libraries(rng_test gtest gtest_main g3logger -lpthread -lm -lcurand_static  -lculibos pcg_random)
    target_compile_definitions(rng_test PRIVATE -DCUDA)
ELSE()
    SET(SOURCES random_numbers_test.cpp 
                ../src/random_numbers.cpp
                ../src/dsmc_utils.cpp)
    cuda_add_executable(rng_test ${SOURCES})
    target_link_libraries(rng_test gtest gtest_main g3logger -lpthread -lm pcg_random)
ENDIF()

add_test(random_number_generation rng_test)

# DISTRIBUTION GENERATION TESTS
IF(CUDA)
    MESSAGE(WARNING "Compiling device distribution generation tests")
    SET(SOURCES ../src/dsmc_utils.cpp
                ../src/dsmc_utils.cu
                ../src/random_numbers.cu
                ../src/magnetic_field.cpp
                ../src/magnetic_field.cu
                ../src/distribution_generation.cu
                distribution_generation_test.cu)
    cuda_add_executable(distribution_generation_test ${SOURCES}
                        OPTIONS -DCUDA)
    FOREACH(defintion ${PARALLELISATION_DEFINITIONS})
        target_compile_definitions(distribution_generation_test PRIVATE ${defintion})
    ENDFOREACH(defintion)
    target_link_libraries(distribution_generation_test gtest gtest_main g3logger -lpthread -lm -lcurand_static  -lculibos pcg_random ${PARALLELISATION_FLAGS})
ELSE()
    SET(SOURCES ../src/dsmc_utils.cpp
                ../src/random_numbers.cpp
                ../src/magnetic_field.cpp
                ../src/distribution_generation.cpp
                distribution_generation_test.cpp)
    add_executable(distribution_generation_test ${SOURCES})
    FOREACH(defintion ${PARALLELISATION_DEFINITIONS})
        target_compile_definitions(distribution_generation_test PRIVATE ${defintion})
    ENDFOREACH(defintion)
    message("complie definitions: ${PARALLELISATION_DEFINITIONS}")
    message("link flags: ${PARALLELISATION_FLAGS}")
    target_link_libraries(distribution_generation_test gtest gtest_main g3logger -lpthread -lm pcg_random ${PARALLELISATION_FLAGS})
ENDIF()

add_test(distribution_generation distribution_generation_test)

# PARTICLE EVOLUTION TESTS
IF(CUDA)
    MESSAGE(WARNING "Compiling device particle evolution tests")
    SET(SOURCES ../src/dsmc_utils.cpp
                ../src/dsmc_utils.cu
                ../src/random_numbers.cu
                ../src/magnetic_field.cpp
                ../src/magnetic_field.cu
                ../src/distribution_generation.cu
                ../src/particle_evolution.cu
                particle_evolution_test.cu)
    cuda_add_executable(particle_evolution_test ${SOURCES}
                        OPTIONS -DCUDA)
    FOREACH(defintion ${PARALLELISATION_DEFINITIONS})
        target_compile_definitions(particle_evolution_test PRIVATE ${defintion})
    ENDFOREACH(defintion)
    target_link_libraries(particle_evolution_test gtest gtest_main g3logger -lpthread -lm -lcurand_static  -lculibos pcg_random ${PARALLELISATION_FLAGS})
ELSE()
    SET(SOURCES ../src/dsmc_utils.cpp
                ../src/random_numbers.cpp
                ../src/magnetic_field.cpp
                ../src/distribution_generation.cpp
                ../src/particle_evolution.cpp
                particle_evolution_test.cpp)
    add_executable(particle_evolution_test ${SOURCES})
    FOREACH(defintion ${PARALLELISATION_DEFINITIONS})
        target_compile_definitions(particle_evolution_test PRIVATE ${defintion})
    ENDFOREACH(defintion)
    message("complie definitions: ${PARALLELISATION_DEFINITIONS}")
    message("link flags: ${PARALLELISATION_FLAGS}")
    target_link_libraries(particle_evolution_test gtest gtest_main g3logger -lpthread -lm pcg_random ${PARALLELISATION_FLAGS})
ENDIF()

add_test(particle_evolution particle_evolution_test)

# SET(SOURCES omp_test.cpp)
# add_executable(omp_test ${SOURCES})
# target_link_libraries(omp_test -fopenmp)
# add_test(omp omp_test)

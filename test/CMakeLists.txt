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
    SET(SOURCES random_numbers_test.cpp ../src/random_numbers.cpp)
    cuda_add_executable(rng_test ${SOURCES})
    target_link_libraries(rng_test gtest gtest_main g3logger -lpthread -lm pcg_random)
    target_compile_definitions(rng_test PRIVATE -DCUDA)
ELSE()
    SET(SOURCES random_numbers_test.cpp ../src/random_numbers.cpp)
    cuda_add_executable(rng_test ${SOURCES})
    target_link_libraries(rng_test gtest gtest_main g3logger -lpthread -lm pcg_random)
ENDIF()

add_test(random_number_generation rng_test)

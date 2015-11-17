#Project cmake
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "bin")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "bin")

cmake_minimum_required (VERSION 2.8)
project (cuda_dsmc)
find_package(CUDA QUIET REQUIRED)

# add_subdirectory( src )

#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS -arch sm_20)

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/cuda_dsmc" "/Developer/NVIDIA/CUDA-7.0/include/" "/usr/local/cuda/include" "/usr/local/include/")
link_directories("${PROJECT_SOURCE_DIR}/lib" "/usr/local/lib/")
SET( SOURCES src/main.cpp src/distribution_generation.cpp src/random_numbers.cpp )

# add_executable(cuda_dsmc ${SOURCES})
cuda_add_executable(cuda_dsmc ${SOURCES})
target_link_libraries(cuda_dsmc pcg_random)

# make run
add_custom_target(run
    COMMAND cuda_dsmc
    DEPENDS cuda_dsmc
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# does the application run
enable_testing()

SET( TEST_SOURCES test/distribtution_generation_test.cpp src/random_numbers.cpp )
add_executable(distribution_generation_test ${TEST_SOURCES})
target_link_libraries(distribution_generation_test pcg_random testu01)

SET( TEST_SOURCES_NIGHTLY test/distribtution_generation_test_nightly.cpp src/random_numbers.cpp )
add_executable(distribution_generation_test_nightly ${TEST_SOURCES_NIGHTLY})
target_link_libraries(distribution_generation_test_nightly pcg_random testu01)

add_test (cuda_dmsc_runs cuda_dsmc)
add_test (distribution_generation distribution_generation_test -s)
add_test (distribution_generation_nightly distribution_generation_test_nightly -s)

# does it have the correct output
# add_test (hello_world_test hello)
# set_tests_properties( hello_world_test
# 	PROPERTIES PASS_REGULAR_EXPRESSION "Hello World!")

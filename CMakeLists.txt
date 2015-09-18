SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ".")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ".")

cmake_minimum_required (VERSION 2.8)
project (cuda_dsmc)
find_package(CUDA QUIET REQUIRED)
cuda_add_executable(hello hello_world.cu)

# does the application run
enable_testing()

add_test (cuda_dmsc_runs hello)

# does it have the correct output
add_test (hello_world_test hello)
set_tests_properties( hello_world_test
	PROPERTIES PASS_REGULAR_EXPRESSION "Hello, World!")

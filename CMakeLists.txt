#Project cmake
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "bin")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "bin")

cmake_minimum_required (VERSION 2.8)
project (cuda_dsmc)
find_package(CUDA QUIET REQUIRED)

# add_subdirectory( src )

set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS "-std=c++11")

set(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

message("${CUDA_TOOLKIT_ROOT_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/cuda_dsmc" "${CUDA_INCLUDE_DIRS}" "~/Downloads/g3log-1.1/src" "/usr/local/include")
link_directories("${PROJECT_SOURCE_DIR}/lib" "/usr/local/lib/" "${CUDA_TOOLKIT_ROOT_DIR}/lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

IF (WIN32)
	SET(SOURCES src/main.cpp src/main.cpp src/distribution_generation.cpp src/random_numbers.cpp src/entropy.c)
    add_executable(cuda_dsmc ${SOURCES})
    target_link_libraries(cuda_dsmc pcg_random g3logger)
ELSE ()
	SET(SOURCES src/main.cpp src/distribution_generation.cpp src/distribution_generation.cu src/random_numbers.cpp src/random_numbers.cu src/entropy.c)
	add_definitions(-DCUDA)
    cuda_add_executable(cuda_dsmc ${SOURCES})
    target_link_libraries(cuda_dsmc pcg_random cudart g3logger)
ENDIF ()



# make run
add_custom_target(run
    COMMAND cuda_dsmc
    DEPENDS cuda_dsmc
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# ============================================================================
# cmake clean command
# ============================================================================
IF (NOT MSVC)
   MESSAGE("\n\n
      *******************************************************************
      Please do 'make clean-cmake' before next cmake generation. 
      It is a good idea to purge your build directory of CMake 
      generated cache files
      *******************************************************************
       ")
   add_custom_target(clean-cmake
      COMMAND ${CMAKE_COMMAND} -P CleanAll.cmake
   )
ENDIF()

# does the application run
enable_testing()

# SET( TEST_SOURCES test/distribution_generation_test.cpp src/random_numbers.cpp )
# add_executable(distribution_generation_test ${TEST_SOURCES})
# target_link_libraries(distribution_generation_test pcg_random testu01)

add_test (cuda_dmsc_runs cuda_dsmc)
#add_test (distribution_generation distribution_generation_test -s)

# does it have the correct output
# add_test (hello_world_test hello)
# set_tests_properties( hello_world_test
# 	PROPERTIES PASS_REGULAR_EXPRESSION "Hello World!")
# 
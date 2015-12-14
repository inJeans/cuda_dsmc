#Project cmake
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "Debug")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "Release")

# Add project cmake modules to path.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

cmake_minimum_required (VERSION 2.8)
project (cuda_dsmc)
IF (NOT WIN32)
    find_package(CUDA QUIET REQUIRED)
    SET(CUDA_NVCC_FLAGS "-std=c++11")
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
ENDIF()

SET(CMAKE_C_FLAGS "-std=c99 ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/cuda_dsmc")
IF (NOT WIN32)
    include_directories("${CUDA_INCLUDE_DIRS}" "/usr/local/include")
ENDIF ()

link_directories("${PROJECT_SOURCE_DIR}/lib")
IF (UNIX)
    link_directories("/usr/local/lib/")
    IF (APPLE)
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")
    ELSE ()
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    ENDIF()
ENDIF ()

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

# We probably don't want this to run on every build.
option(COVERALLS "Generate coveralls data" OFF)

IF (COVERALLS)
    include(cmake/Coveralls.cmake)
    coveralls_turn_on_coverage()
    SET(COVERAGE_SRCS "${SOURCES}")

    # Create the coveralls target.
    coveralls_setup(
        "${COVERAGE_SRCS}" # The source files.
        ON)                # If we should upload.
ENDIF()

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
      COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/CleanAll.cmake
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
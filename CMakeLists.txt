#Project cmake
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "Debug")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "Release")

# The project version number.
set(VERSION_MAJOR   2   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

# Add project cmake modules to path.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project (cuda_dsmc)

# User defined g3log library destintation
IF(G3LOG_LIBRARY_PATH)
    link_directories("${G3LOG_LIBRARY_PATH}")
ENDIF()

# User defined g3log header destintation
IF(G3LOG_INCLUDE_PATH)
    message(STATUS "G3LOG_INCLUDE_PATH='${G3LOG_INCLUDE_PATH}'")
    include_directories("${G3LOG_INCLUDE_PATH}")
ENDIF()

# User defined pcg library destintation
IF(PCG_LIBRARY_PATH)
    link_directories("${PCG_LIBRARY_PATH}")
ENDIF()

# User defined pcg header destintation
IF(PCG_INCLUDE_PATH)
    include_directories("${PCG_INCLUDE_PATH}")
ENDIF()

# User defined omp library destintation
IF(OMP_LIBRARY_PATH)
    # set(OPENMP_C_LIBRARIES "${OMP_LIBRARY_PATH}")
    # set(OPENMP_CXX_LIBRARIES "${OMP_LIBRARY_PATH}")
    link_directories("${OMP_LIBRARY_PATH}")
ENDIF()

# User defined omp header destintation
IF(OMP_INCLUDE_PATH)
    # set(OPENMP_C_INCLUDES "${OMP_INCLUDE_PATH}")
    # set(OPENMP_CXX_INCLUDES "${OMP_INCLUDE_PATH}")
    include_directories("${OMP_INCLUDE_PATH}")
ENDIF()

SET(PARALLELISATION_DEFINITIONS CACHE INTERNAL "Variable for holding parallelisation definition")
SET(PARALLELISATION_FLAGS CACHE INTERNAL "Variable for holding parallelisation flags")

# We will optionally build with CUDA.
option(CUDA "Compile with GPU acceleration" OFF)

IF(NOT WIN32)
    find_package(CUDA REQUIRED)
    SET(CUDA_NVCC_FLAGS "-std=c++11")
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
ENDIF()

SET(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/include"
                    "${PROJECT_SOURCE_DIR}/include/cuda_dsmc"
                    "/usr/local/include"
                    "/usr/local/")
IF(CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
ENDIF()

IF(UNIX)
    link_directories("/usr/local/lib/")
    IF(APPLE)
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")
    ELSE()
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    ENDIF()
ENDIF()

# We will optionally build with MPI.
option(MPI "Compile with MPI distribution" OFF)

IF(MPI)
  find_package(MPI REQUIRED)
  LIST(APPEND PARALLELISATION_DEFINITIONS "DSMC_MPI")
  LIST(APPEND PARALLELISATION_FLAGS "-lmpi")
ENDIF()

# We will optionally build with OpenMP.
option(OPENMP "Compile with OpenMP parallelism" OFF)

IF(OPENMP)
  # IF(APPLE)
  #   message("${CMAKE_C_COMPILER_ID}")
  #   IF(CMAKE_C_COMPILER_ID MATCHES "Clang")
  #       SET(OpenMP_C "${CMAKE_C_COMPILER}")
  #       SET(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
  #       SET(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
  #       SET(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
  #       SET(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
  #       SET(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
  #   ENDIF()
  #   message("${OpenMP_C_FLAGS}")
  #   message("${CMAKE_CXX_COMPILER_ID}")
  #   IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  #     SET(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
  #     SET(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
  #     SET(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
  #     SET(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
  #     SET(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
  #     SET(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
  #   ENDIF()
  #   message("${OpenMP_CXX_FLAGS}")
  # ENDIF()
  LIST(APPEND PARALLELISATION_DEFINITIONS "OPENMP")
  IF (APPLE)
    LIST(APPEND PARALLELISATION_FLAGS "-fopenmp")
  ELSE()
    find_package(OPENMP REQUIRED)
    LIST(APPEND PARALLELISATION_FLAGS "-lomp")
  ENDIF()
  # find_package(OpenMP)
#   if (OPENMP_FOUND)
#     include_directories("${OPENMP_INCLUDE_PATH}")
#     link_directories("${OPENMP_LIBRARY_PATH}")
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     # set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  # endif(OPENMP_FOUND)
ENDIF()

# We probably don't want this to run on every build.
# option(COVERALLS "Generate coveralls data" OFF)

# IF(COVERALLS)
#     include(cmake/Coveralls.cmake)
#     coveralls_turn_on_coverage()
#     SET(COVERAGE_SRCS "${SOURCES}")

#     # Create the coveralls target.
#     coveralls_setup(
#         "${COVERAGE_SRCS}" # The source files.
#         ON)                # If we should upload.
# ENDIF()

# ============================================================================
# cmake clean command
# ============================================================================
IF(NOT MSVC)
   MESSAGE("\n
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

enable_testing()
# Compile source
add_subdirectory(src)
# Compile tests
add_subdirectory(test)
# Compile docs
add_subdirectory(docs)

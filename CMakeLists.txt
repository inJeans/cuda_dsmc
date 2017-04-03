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

# We will optionally build with CUDA.
option(CUDA "Compile with GPU acceleration" OFF)

IF(NOT WIN32)
    find_package(CUDA QUIET REQUIRED)
    SET(CUDA_NVCC_FLAGS "-std=c++11")
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
ENDIF()

SET(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGSw11}")
SET(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/cuda_dsmc")
IF(CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}" "/usr/local/include")
ENDIF()

IF(UNIX)
    link_directories("/usr/local/lib/")
    IF(APPLE)
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")
    ELSE()
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    ENDIF()
ENDIF()

# We probably don't want this to run on every build.
option(COVERALLS "Generate coveralls data" OFF)

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
add_subdirectory(doc)

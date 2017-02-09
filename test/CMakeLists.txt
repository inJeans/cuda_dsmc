cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project (cuda_dsmc_test)

# User defined testu01 library destintation
IF(TESTU01_LIBRARY_PATH)
    link_directories("${TESTU01_LIBRARY_PATH}")
ENDIF()

# User defined testu01 header destintation
IF(TESTU01_INCLUDE_PATH)
    include_directories("${TESTU01_INCLUDE_PATH}")
ENDIF()

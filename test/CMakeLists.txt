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

IF(CUDA)
    # add_definitions(-DCUDA)
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

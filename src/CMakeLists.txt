cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project (cuda_dsmc_exec)

IF(CUDA)
    # add_definitions(-DCUDA)
    SET(SOURCES main.cpp )
    cuda_add_executable(cuda_dsmc ${SOURCES})
    target_link_libraries(cuda_dsmc)
    target_compile_definitions(cuda_dsmc PRIVATE -DCUDA)
ELSE()
    SET(SOURCES main.cpp)
    cuda_add_executable(cuda_dsmc ${SOURCES})
    target_link_libraries(cuda_dsmc)
ENDIF()

target_link_libraries(cuda_dsmc g3logger)

# make run
add_custom_target(run
    COMMAND cuda_dsmc
    DEPENDS cuda_dsmc
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

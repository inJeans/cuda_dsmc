cmake_minimum_required (VERSION 2.6)
project (cuda_dsmc)
add_executable(hello hello_world.c)

# does the application run
add_test (cuda_dmsc_runs hello)

# does it have the correct output
add_test (hello_world_test hello)
set_tests_properties( hello_world_test
	PROPERTIES PASS_REGULAR_EXPRESSION "Hello, World!")

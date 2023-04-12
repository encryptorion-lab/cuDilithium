# test source files
set(TEST_SRCS test_vectors.cu p_randombytes.c)

# Dilithium 2 test vectors
add_executable(test_cuDilithium2 ${TEST_SRCS})
target_compile_definitions(test_cuDilithium2 PRIVATE DILITHIUM_MODE=2)
target_link_libraries(test_cuDilithium2 cuDilithium2)

# Dilithium 3 test vectors
add_executable(test_cuDilithium3 ${TEST_SRCS})
target_compile_definitions(test_cuDilithium3 PRIVATE DILITHIUM_MODE=3)
target_link_libraries(test_cuDilithium3 cuDilithium3)

# Dilithium 5 test vectors
add_executable(test_cuDilithium5 ${TEST_SRCS})
target_compile_definitions(test_cuDilithium5 PRIVATE DILITHIUM_MODE=5)
target_link_libraries(test_cuDilithium5 cuDilithium5)

# add CTests for test vectors
# we cut off the official test vectors to only check m,pk,sk,sig
add_test(NAME test_cuDilithium2 COMMAND sh -c "\"$<TARGET_FILE:test_cuDilithium2>\" && echo \"a0835b0f41084881409e42866012996d2bf72fb8694535f190b2a2bfaf271660 tvecs2\" | sha256sum --check")
add_test(NAME test_cuDilithium3 COMMAND sh -c "\"$<TARGET_FILE:test_cuDilithium3>\" && echo \"98c70f95be840482fd656b264a559fa770c0e693629d019504b42c8229bab123 tvecs3\" | sha256sum --check")
add_test(NAME test_cuDilithium5 COMMAND sh -c "\"$<TARGET_FILE:test_cuDilithium5>\" && echo \"d702f8ae15b5d9f0234120a3a073e617f7aebc9d52df0beae7bb47e14a17c1d3 tvecs5\" | sha256sum --check")

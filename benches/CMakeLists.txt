# bench source files
set(BENCH_SRCS bench_dilithium.cu trng.c)

# cuDilithium2 bench executable
add_executable(bench_cuDilithium2 ${BENCH_SRCS})
target_compile_definitions(bench_cuDilithium2 PRIVATE DILITHIUM_MODE=2)
target_link_libraries(bench_cuDilithium2 cuDilithium2)

# cuDilithium3 bench executable
add_executable(bench_cuDilithium3 ${BENCH_SRCS})
target_compile_definitions(bench_cuDilithium3 PRIVATE DILITHIUM_MODE=3)
target_link_libraries(bench_cuDilithium3 cuDilithium3)

# cuDilithium5 bench executable
add_executable(bench_cuDilithium5 ${BENCH_SRCS})
target_compile_definitions(bench_cuDilithium5 PRIVATE DILITHIUM_MODE=5)
target_link_libraries(bench_cuDilithium5 cuDilithium5)

# radix-2 NTT bench executable
add_executable(bench_ntt_radix2 bench_ntt_radix2.cu)
target_compile_definitions(bench_ntt_radix2 PRIVATE DILITHIUM_MODE=2)

# radix-8 NTT bench executable
add_executable(bench_ntt_radix8 bench_ntt_radix8.cu)
target_compile_definitions(bench_ntt_radix8 PRIVATE DILITHIUM_MODE=2)

# Reject sampling bench executable
add_executable(bench_rej_sampling bench_rej_sampling.cu)
target_compile_definitions(bench_rej_sampling PRIVATE DILITHIUM_MODE=2)

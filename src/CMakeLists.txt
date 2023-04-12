set(DILITHIUM_SRCS ntt.cu api.cu keypair.cu sign.cu verify.cu)

# Dilithium 2 library
add_library(cuDilithium2 ${DILITHIUM_SRCS})
target_compile_definitions(cuDilithium2 PRIVATE DILITHIUM_MODE=2)
target_link_libraries(cuDilithium2 INTERFACE fips202)

# Dilithium 3 library
add_library(cuDilithium3 ${DILITHIUM_SRCS})
target_compile_definitions(cuDilithium3 PRIVATE DILITHIUM_MODE=3)
target_link_libraries(cuDilithium3 INTERFACE fips202)

# Dilithium 5 library
add_library(cuDilithium5 ${DILITHIUM_SRCS})
target_compile_definitions(cuDilithium5 PRIVATE DILITHIUM_MODE=5)
target_link_libraries(cuDilithium5 INTERFACE fips202)

add_subdirectory(fips202)

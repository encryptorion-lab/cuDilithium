#pragma once

#include <cstdint>

__global__ void gpu_keypair(uint8_t *g_pk_rho, uint8_t *g_pk_t1_packed,
                            uint8_t *g_sk_rho, uint8_t *g_sk_key, uint8_t *g_sk_tr,
                            uint8_t *g_sk_s1_packed, uint8_t *g_sk_s2_packed, uint8_t *g_sk_t0_packed,
                            int32_t *g_t1, size_t keypair_mem_pool_pitch, size_t rand_index = 0);

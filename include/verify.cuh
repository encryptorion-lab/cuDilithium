#pragma once

#include <cstdint>

__global__ void gpu_verify(int *g_ret,
                           const uint8_t *g_c, const uint8_t *g_z_packed, const uint8_t *g_h_packed,
                           const uint8_t *g_rho, const uint8_t *g_t1_packed,
                           uint8_t *g_muprime, const uint8_t *g_m, uint8_t *g_mu, uint8_t *g_w1prime_packed, uint8_t *g_ctilde,
                           int32_t *g_mat, int32_t *g_z, int32_t *g_t1, int32_t *g_w1prime, int32_t *g_h, int32_t *g_cp,
                           size_t mlen, size_t mem_pool_pitch);

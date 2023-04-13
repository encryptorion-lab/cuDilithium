/*
 * Copyright 2022-2023 [Anonymous].
 *
 * This file is part of cuDilithium.
 *
 * cuDilithium is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * cuDilithium is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with cuDilithium.
 * If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <cstdint>

__global__ void gpu_verify(int *g_ret,
                           const uint8_t *g_c, const uint8_t *g_z_packed, const uint8_t *g_h_packed,
                           const uint8_t *g_rho, const uint8_t *g_t1_packed,
                           uint8_t *g_muprime, const uint8_t *g_m, uint8_t *g_mu, uint8_t *g_w1prime_packed, uint8_t *g_ctilde,
                           int32_t *g_mat, int32_t *g_z, int32_t *g_t1, int32_t *g_w1prime, int32_t *g_h, int32_t *g_cp,
                           size_t mlen, size_t mem_pool_pitch);

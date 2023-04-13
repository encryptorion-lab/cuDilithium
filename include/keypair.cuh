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

__global__ void gpu_keypair(uint8_t *g_pk_rho, uint8_t *g_pk_t1_packed,
                            uint8_t *g_sk_rho, uint8_t *g_sk_key, uint8_t *g_sk_tr,
                            uint8_t *g_sk_s1_packed, uint8_t *g_sk_s2_packed, uint8_t *g_sk_t0_packed,
                            int32_t *g_t1, size_t keypair_mem_pool_pitch, size_t rand_index = 0);

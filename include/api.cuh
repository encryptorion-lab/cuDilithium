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

#include <cstddef>
#include <cstdint>

#include "params.h"
#include "util.cuh"

#define ALIGN_TO_256_BYTES(x) ((((x) + 255) / 256) * 256)

int crypto_sign_keypair(uint8_t *pk, uint8_t *sk,
                        uint8_t *d_keypair_mem_pool, size_t keypair_mem_pool_pitch,
                        size_t batch_size = 1, cudaStream_t stream = nullptr, size_t rand_index = 0);

struct sign_lut_element {
    int done_flag = 0;
    uint16_t nonce = 0;
};

struct copy_lut_element {
    size_t exec_idx;
    size_t sign_idx;
};

struct task_lut {
    std::vector<sign_lut_element> sign_lut;
    uint32_t *h_exec_lut;
    uint8_t *h_done_lut;
    copy_lut_element *h_copy_lut;
    uint32_t *d_exec_lut;
    uint8_t *d_done_lut;
    copy_lut_element *d_copy_lut;
};

int crypto_sign_signature(uint8_t *sig, size_t sig_pitch, size_t *siglen,
                          const uint8_t *m, size_t m_pitch, size_t mlen,
                          const uint8_t *sk,
                          uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch,
                          uint8_t *d_temp_mem_pool, size_t d_temp_mem_pool_pitch,
                          task_lut &lut, size_t exec_threshold, size_t batch_size = 1, cudaStream_t stream = nullptr);

void crypto_sign_verify(int *ret,
                        const uint8_t *sig, size_t sig_pitch, size_t siglen,
                        const uint8_t *m, size_t m_pitch, size_t mlen,
                        const uint8_t *pk,
                        uint8_t *d_verify_mem_pool, size_t verify_mem_pool_pitch,
                        size_t batch_size = 1, cudaStream_t stream = nullptr);

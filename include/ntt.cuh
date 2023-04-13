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

#include "params.h"
#include "util.cuh"

extern __device__ const int32_t c_zetas[DILITHIUM_N];

__device__ void ntt_inner(int32_t regs[8], int32_t *s_poly);

__device__ void invntt_inner(int32_t regs[8], int32_t *s_poly);

template<unsigned int VEC_SIZE>
__global__ void ntt_radix8_kernel(int32_t *g_polyvec, size_t mem_pool_pitch) {
    __shared__ int32_t s_poly[DILITHIUM_N + 32];
    int32_t regs[8];
#pragma unroll
    for (int vec_i = 0; vec_i < VEC_SIZE; vec_i++) {
        int32_t *g_poly = g_polyvec + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + vec_i * DILITHIUM_N;
#pragma unroll
        for (size_t i = 0; i < 8; i++)
            regs[i] = g_poly[i * 32 + threadIdx.x];
        ntt_inner(regs, s_poly);
#pragma unroll
        for (size_t i = 0; i < 8; i++)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

template<unsigned int VEC_SIZE>
__global__ void intt_radix8_kernel(int32_t *g_polyvec, size_t mem_pool_pitch) {
    __shared__ int32_t s_poly[DILITHIUM_N + 32];
    int32_t regs[8];
#pragma unroll
    for (int vec_i = 0; vec_i < VEC_SIZE; vec_i++) {
        int32_t *g_poly = g_polyvec + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + vec_i * DILITHIUM_N;
#pragma unroll
        for (size_t i = 0; i < 8; i++)
            regs[i] = g_poly[threadIdx.x * 8 + i];
        invntt_inner(regs, s_poly);
#pragma unroll
        for (size_t i = 0; i < 8; i++)
            g_poly[i * 32 + threadIdx.x] = regs[i];
    }
}

__device__ __forceinline__ int32_t gpu_montgomery_multiply(int32_t x, int32_t y) {
#define QINV 58728449// q^(-1) mod 2^32
    int32_t a_hi = __mulhi(x, y);
    int32_t a_lo = x * y;
    int32_t t = a_lo * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = a_hi - t;
    return t;
}

__device__ __forceinline__ void
ntt_radix2_inner(int32_t &reg0, int32_t &reg1, int32_t s_ntt[DILITHIUM_N + 128], const int32_t s_zetas[DILITHIUM_N]) {
    size_t butt_idx;
    int32_t t;
    int32_t zeta;

    // level 1
    t = reg1 * 25847 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, 25847) - t;
    s_ntt[threadIdx.x + 128] = reg0 - t;
    s_ntt[threadIdx.x] = reg0 + t;
    // store
    __syncthreads();

    // level 2
    butt_idx = (threadIdx.x & 0xFFFFFFC0) + threadIdx.x;
    reg1 = s_ntt[butt_idx + 64];
    zeta = s_zetas[2 + (threadIdx.x >> 6)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx];
    s_ntt[butt_idx + 64] = reg0 - t;
    s_ntt[butt_idx] = reg0 + t;
    __syncthreads();

    // level 3
    butt_idx = (threadIdx.x & 0xFFFFFFE0) + threadIdx.x;
    reg1 = s_ntt[butt_idx + 32];
    zeta = s_zetas[4 + (threadIdx.x >> 5)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx];
    __syncthreads();
    butt_idx += ((butt_idx >> 5) << 4);
    s_ntt[butt_idx + 32 + 16] = reg0 - t;
    s_ntt[butt_idx] = reg0 + t;

    // level 4
    butt_idx = (threadIdx.x & 0xFFFFFFF0) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 5) << 4) + 16];
    zeta = s_zetas[8 + (threadIdx.x >> 4)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 5) << 4)];
    s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 16 + 8] = reg0 - t;
    s_ntt[butt_idx + ((butt_idx >> 4) << 3)] = reg0 + t;

    // level 5
    butt_idx = (threadIdx.x & 0xFFFFFFF8) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 8];
    zeta = s_zetas[16 + (threadIdx.x >> 3)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 4) << 3)];
    s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 8 + 4] = reg0 - t;
    s_ntt[butt_idx + ((butt_idx >> 3) << 2)] = reg0 + t;

    // level 6
    butt_idx = (threadIdx.x & 0xFFFFFFFC) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 4];
    zeta = s_zetas[32 + (threadIdx.x >> 2)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 3) << 2)];
    s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 4 + 2] = reg0 - t;
    s_ntt[butt_idx + ((butt_idx >> 2) << 1)] = reg0 + t;

    // level 7
    butt_idx = (threadIdx.x & 0xFFFFFFFE) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 2];
    zeta = s_zetas[64 + (threadIdx.x >> 1)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 2) << 1)];
    s_ntt[butt_idx + (butt_idx >> 1) + 2 + 1] = reg0 - t;
    s_ntt[butt_idx + (butt_idx >> 1)] = reg0 + t;

    // level 8
    reg1 = s_ntt[3 * threadIdx.x + 1];
    zeta = s_zetas[128 + threadIdx.x];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[3 * threadIdx.x];
    __syncthreads();
    reg1 = reg0 - t;
    reg0 = reg0 + t;

    // permute index
    s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4)] = reg0;
    s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4) + 1] = reg1;
    __syncthreads();
    reg0 = s_ntt[threadIdx.x + (threadIdx.x >> 5)];
    reg1 = s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)];
}

__device__ __forceinline__ void
intt_radix2_inner(int32_t &reg0, int32_t &reg1, int32_t s_ntt[DILITHIUM_N + 128], const int32_t s_zetas[DILITHIUM_N]) {
#define MONT2DIVN 41978
#define MONT2DIVNMULZETA 3975713
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // permute index
    s_ntt[threadIdx.x + (threadIdx.x >> 5)] = reg0;
    s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] = reg1;
    __syncthreads();
    reg0 = s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4)];
    reg1 = s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4) + 1];

    // level 8
    butt_idx = threadIdx.x + threadIdx.x;
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, -s_zetas[255 - threadIdx.x]);
    __syncthreads();
    // store
    s_ntt[butt_idx + (butt_idx >> 1)] = reg0;
    s_ntt[butt_idx + (butt_idx >> 1) + 1] = reg1;

    // level 7
    w_idx = threadIdx.x >> 1;
    butt_idx = (w_idx << 1) + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + (butt_idx >> 1)];
    reg1 = s_ntt[butt_idx + (butt_idx >> 1) + 2 + 1];
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, -s_zetas[127 - w_idx]);
    // store
    s_ntt[butt_idx + ((butt_idx >> 2) << 1)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 2] = reg1;

    // level 6
    w_idx = threadIdx.x >> 2;
    butt_idx = (w_idx << 2) + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + ((butt_idx >> 2) << 1)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 4 + 2];
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, -s_zetas[63 - w_idx]);
    // store
    s_ntt[butt_idx + ((butt_idx >> 3) << 2)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 4] = reg1;

    // level 5
    w_idx = threadIdx.x >> 3;
    butt_idx = (w_idx << 3) + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + ((butt_idx >> 3) << 2)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 8 + 4];
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, -s_zetas[31 - w_idx]);
    // store
    s_ntt[butt_idx + ((butt_idx >> 4) << 3)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 8] = reg1;

    // level 4
    w_idx = threadIdx.x >> 4;
    butt_idx = (w_idx << 4) + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + ((butt_idx >> 4) << 3)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 16 + 8];
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, -s_zetas[15 - w_idx]);
    // store
    s_ntt[butt_idx + ((butt_idx >> 5) << 4)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 5) << 4) + 16] = reg1;

    // level 3
    w_idx = threadIdx.x >> 5;
    butt_idx = (w_idx << 5) + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + ((butt_idx >> 5) << 4)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 5) << 4) + 32 + 16];
    __syncthreads();
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, -s_zetas[7 - w_idx]);
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 32] = reg1;
    __syncthreads();

    // level 2
    w_idx = threadIdx.x >> 6;
    butt_idx = (w_idx << 6) + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 64];
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, -s_zetas[3 - w_idx]);
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 64] = reg1;
    __syncthreads();

    // level 1
    // load
    t = s_ntt[threadIdx.x];
    reg1 = s_ntt[threadIdx.x + 128];
    // compute
    reg0 = t + reg1;
    reg0 = gpu_montgomery_multiply(reg0, MONT2DIVN);
    reg1 = t - reg1;
    reg1 = gpu_montgomery_multiply(reg1, MONT2DIVNMULZETA);
}

template<unsigned int VEC_SIZE>
__global__ void ntt_radix2_kernel(int32_t *g_polyvec, size_t mem_pool_pitch) {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];

    s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
    s_zetas[threadIdx.x + 128] = c_zetas[threadIdx.x + 128];
    __syncthreads();

    for (int vec_i = 0; vec_i < VEC_SIZE; vec_i++) {
        int32_t *g_poly = g_polyvec + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + vec_i * DILITHIUM_N;

        int32_t reg0 = g_poly[threadIdx.x];
        int32_t reg1 = g_poly[threadIdx.x + 128];

        ntt_radix2_inner(reg0, reg1, s_ntt, s_zetas);

        g_poly[threadIdx.x] = reg0;
        g_poly[threadIdx.x + 128] = reg1;
    }
}

__device__ __forceinline__ void ntt_radix2_inner_pad1(int32_t &reg0, int32_t &reg1, int32_t s_ntt[DILITHIUM_N + 128],
                                                      const int32_t s_zetas[DILITHIUM_N]) {
    size_t butt_idx;
    int32_t t;
    int32_t zeta;

    // level 1
    t = reg1 * 25847 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, 25847) - t;
    s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] = reg0 - t;
    s_ntt[threadIdx.x + (threadIdx.x >> 5)] = reg0 + t;
    // store
    __syncthreads();

    // level 2
    butt_idx = (threadIdx.x & 0xFFFFFFC0) + threadIdx.x;
    butt_idx += (butt_idx >> 5);
    reg1 = s_ntt[butt_idx + 64 + 2];
    zeta = s_zetas[2 + (threadIdx.x >> 6)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx];
    s_ntt[butt_idx + 64 + 2] = reg0 - t;
    s_ntt[butt_idx] = reg0 + t;
    __syncthreads();

    // level 3
    butt_idx = (threadIdx.x & 0xFFFFFFE0) + threadIdx.x;
    reg1 = s_ntt[butt_idx + (butt_idx >> 5) + 32 + 1];
    zeta = s_zetas[4 + (threadIdx.x >> 5)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + (butt_idx >> 5)];
    __syncthreads();
    butt_idx += ((butt_idx >> 5) << 4);
    s_ntt[butt_idx + 32 + 16] = reg0 - t;
    s_ntt[butt_idx] = reg0 + t;

    // level 4
    butt_idx = (threadIdx.x & 0xFFFFFFF0) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 5) << 4) + 16];
    zeta = s_zetas[8 + (threadIdx.x >> 4)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 5) << 4)];
    s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 16 + 8] = reg0 - t;
    s_ntt[butt_idx + ((butt_idx >> 4) << 3)] = reg0 + t;

    // level 5
    butt_idx = (threadIdx.x & 0xFFFFFFF8) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 8];
    zeta = s_zetas[16 + (threadIdx.x >> 3)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 4) << 3)];
    s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 8 + 4] = reg0 - t;
    s_ntt[butt_idx + ((butt_idx >> 3) << 2)] = reg0 + t;

    // level 6
    butt_idx = (threadIdx.x & 0xFFFFFFFC) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 4];
    zeta = s_zetas[32 + (threadIdx.x >> 2)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 3) << 2)];
    s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 4 + 2] = reg0 - t;
    s_ntt[butt_idx + ((butt_idx >> 2) << 1)] = reg0 + t;

    // level 7
    butt_idx = (threadIdx.x & 0xFFFFFFFE) + threadIdx.x;
    reg1 = s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 2];
    zeta = s_zetas[64 + (threadIdx.x >> 1)];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 2) << 1)];
    s_ntt[butt_idx + (butt_idx >> 1) + 2 + 1] = reg0 - t;
    s_ntt[butt_idx + (butt_idx >> 1)] = reg0 + t;

    // level 8
    reg1 = s_ntt[3 * threadIdx.x + 1];
    zeta = s_zetas[128 + threadIdx.x];
    t = reg1 * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, zeta) - t;
    reg0 = s_ntt[3 * threadIdx.x];
    __syncthreads();
    reg1 = reg0 - t;
    reg0 = reg0 + t;

    // permute index
    s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4)] = reg0;
    s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4) + 1] = reg1;
    __syncthreads();
    reg0 = s_ntt[threadIdx.x + (threadIdx.x >> 5)];
    reg1 = s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)];
}

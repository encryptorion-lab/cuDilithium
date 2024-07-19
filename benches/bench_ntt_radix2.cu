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

#include "params.h"
#include "util.cuh"

#define NTESTS 10000

#define QINV 58728449// q^(-1) mod 2^32
#define MONT (-4186625)
#define MONT2DIVN 41978
#define MONT2DIVNMULZETA 3975713

__device__ static const int32_t c_zetas[DILITHIUM_N] = {
        0, 25847, -2608894, -518909, 237124, -777960, -876248, 466468,
        1826347, 2353451, -359251, -2091905, 3119733, -2884855, 3111497, 2680103,
        2725464, 1024112, -1079900, 3585928, -549488, -1119584, 2619752, -2108549,
        -2118186, -3859737, -1399561, -3277672, 1757237, -19422, 4010497, 280005,
        2706023, 95776, 3077325, 3530437, -1661693, -3592148, -2537516, 3915439,
        -3861115, -3043716, 3574422, -2867647, 3539968, -300467, 2348700, -539299,
        -1699267, -1643818, 3505694, -3821735, 3507263, -2140649, -1600420, 3699596,
        811944, 531354, 954230, 3881043, 3900724, -2556880, 2071892, -2797779,
        -3930395, -1528703, -3677745, -3041255, -1452451, 3475950, 2176455, -1585221,
        -1257611, 1939314, -4083598, -1000202, -3190144, -3157330, -3632928, 126922,
        3412210, -983419, 2147896, 2715295, -2967645, -3693493, -411027, -2477047,
        -671102, -1228525, -22981, -1308169, -381987, 1349076, 1852771, -1430430,
        -3343383, 264944, 508951, 3097992, 44288, -1100098, 904516, 3958618,
        -3724342, -8578, 1653064, -3249728, 2389356, -210977, 759969, -1316856,
        189548, -3553272, 3159746, -1851402, -2409325, -177440, 1315589, 1341330,
        1285669, -1584928, -812732, -1439742, -3019102, -3881060, -3628969, 3839961,
        2091667, 3407706, 2316500, 3817976, -3342478, 2244091, -2446433, -3562462,
        266997, 2434439, -1235728, 3513181, -3520352, -3759364, -1197226, -3193378,
        900702, 1859098, 909542, 819034, 495491, -1613174, -43260, -522500,
        -655327, -3122442, 2031748, 3207046, -3556995, -525098, -768622, -3595838,
        342297, 286988, -2437823, 4108315, 3437287, -3342277, 1735879, 203044,
        2842341, 2691481, -2590150, 1265009, 4055324, 1247620, 2486353, 1595974,
        -3767016, 1250494, 2635921, -3548272, -2994039, 1869119, 1903435, -1050970,
        -1333058, 1237275, -3318210, -1430225, -451100, 1312455, 3306115, -1962642,
        -1279661, 1917081, -2546312, -1374803, 1500165, 777191, 2235880, 3406031,
        -542412, -2831860, -1671176, -1846953, -2584293, -3724270, 594136, -3776993,
        -2013608, 2432395, 2454455, -164721, 1957272, 3369112, 185531, -1207385,
        -3183426, 162844, 1616392, 3014001, 810149, 1652634, -3694233, -1799107,
        -3038916, 3523897, 3866901, 269760, 2213111, -975884, 1717735, 472078,
        -426683, 1723600, -1803090, 1910376, -1667432, -1104333, -260646, -3833893,
        -2939036, -2235985, -420899, -2286327, 183443, -976891, 1612842, -3545687,
        -554416, 3919660, -48306, -1362209, 3937738, 1400424, -846154, 1976782};

__device__ __inline__ static int32_t montgomery_multiply(int32_t x, int32_t y) {
    int32_t t;

    asm(
            "{\n\t"
            " .reg .s32 a_hi, a_lo;\n\t"
            " mul.hi.s32 a_hi, %1, %2;\n\t"
            " mul.lo.s32 a_lo, %1, %2;\n\t"
            " mul.lo.s32 %0, a_lo, %4;\n\t"
            " mul.hi.s32 %0, %0, %3;\n\t"
            " sub.s32 %0, a_hi, %0;\n\t"
            "}"
            : "=r"(t)
            : "r"(x), "r"(y), "r"(DILITHIUM_Q), "r"(QINV));
    return t;
}

__device__ __inline__ static int32_t montgomery_multiply_c(int32_t x, int32_t y) {
    int32_t a_hi = __mulhi(x, y);
    int32_t a_lo = x * y;
    int32_t t = a_lo * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = a_hi - t;
    return t;
}

__device__ static void ntt_butt(int32_t &a, int32_t &b, const int32_t zeta) {
    int32_t t = montgomery_multiply(zeta, b);
    b = a - t;
    a = a + t;
}

__device__ static void invntt_butt(int32_t &a, int32_t &b, const int32_t zeta) {
    int32_t t = a;
    a = t + b;
    b = t - b;
    b = montgomery_multiply(zeta, b);
}

__device__ void ntt_radix2_inner_opt0(int32_t s_ntt[DILITHIUM_N]) {
    for (size_t log_m = 0; log_m < 8; log_m++) {
        size_t log_step = 7 - log_m;
        size_t w_idx = threadIdx.x >> log_step;
        size_t butt_idx = (w_idx << log_step) + threadIdx.x;
        ntt_butt(s_ntt[butt_idx], s_ntt[butt_idx + (1 << log_step)],
                 c_zetas[(1 << log_m) + w_idx]);
        __syncthreads();
    }
}

__device__ void intt_radix2_inner_opt0(int32_t s_ntt[DILITHIUM_N]) {
    for (int log_m = 7; log_m >= 0; log_m--) {
        size_t log_step = 7 - log_m;
        size_t w_idx = threadIdx.x >> log_step;
        size_t butt_idx = (w_idx << log_step) + threadIdx.x;
        invntt_butt(s_ntt[butt_idx], s_ntt[butt_idx + (1 << log_step)],
                    -c_zetas[(1 << (log_m + 1)) - 1 - w_idx]);
        __syncthreads();
    }
    s_ntt[threadIdx.x] = montgomery_multiply(s_ntt[threadIdx.x], MONT2DIVN);
    s_ntt[threadIdx.x + 128] = montgomery_multiply(s_ntt[threadIdx.x + 128], MONT2DIVN);
    __syncthreads();
}

// unroll
__device__ void ntt_radix2_inner_opt1(int32_t s_ntt[DILITHIUM_N]) {
    int32_t reg0, reg1;
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // level 1
    // load
    reg0 = s_ntt[threadIdx.x];
    reg1 = s_ntt[threadIdx.x + 128];
    // compute
    t = montgomery_multiply(reg1, c_zetas[1]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[threadIdx.x] = reg0;
    s_ntt[threadIdx.x + 128] = reg1;
    __syncthreads();

    // level 2
    // load
    w_idx = threadIdx.x >> 6;
    butt_idx = (w_idx << 6) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 64];
    // compute
    t = montgomery_multiply(reg1, c_zetas[2 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 64] = reg1;
    __syncthreads();

    // level 3
    // load
    w_idx = threadIdx.x >> 5;
    butt_idx = (w_idx << 5) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 32];
    // compute
    t = montgomery_multiply(reg1, c_zetas[4 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 32] = reg1;
    __syncthreads();

    // level 4
    // load
    w_idx = threadIdx.x >> 4;
    butt_idx = (w_idx << 4) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 16];
    // compute
    t = montgomery_multiply(reg1, c_zetas[8 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 16] = reg1;
    __syncthreads();

    // level 5
    // load
    w_idx = threadIdx.x >> 3;
    butt_idx = (w_idx << 3) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 8];
    // compute
    t = montgomery_multiply(reg1, c_zetas[16 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 8] = reg1;
    __syncthreads();

    // level 6
    // load
    w_idx = threadIdx.x >> 2;
    butt_idx = (w_idx << 2) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 4];
    // compute
    t = montgomery_multiply(reg1, c_zetas[32 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 4] = reg1;
    __syncthreads();

    // level 7
    // load
    w_idx = threadIdx.x >> 1;
    butt_idx = (w_idx << 1) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 2];
    // compute
    t = montgomery_multiply(reg1, c_zetas[64 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 2] = reg1;
    __syncthreads();

    // level 8
    // load
    butt_idx = threadIdx.x + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 1];
    // compute
    t = montgomery_multiply(reg1, c_zetas[128 + threadIdx.x]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 1] = reg1;
    __syncthreads();
}

// solve bank conflict: two way bank conflict, padding 16 after each row
// pad 16 per 32 coeffs at level 3-4
// pad 8 per 16 coeffs at level 4-5
// pad 4 per coeffs at level 5-6
// pad 2 per coeffs at level 6-7
// pad 1 per coeffs at level 7-8
__device__ void ntt_radix2_inner_opt2(int32_t s_ntt[DILITHIUM_N + 128]) {
    int32_t reg0, reg1;
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // level 1
    // load
    reg0 = s_ntt[threadIdx.x];
    reg1 = s_ntt[threadIdx.x + 128];
    // compute
    t = montgomery_multiply(reg1, c_zetas[1]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[threadIdx.x] = reg0;
    s_ntt[threadIdx.x + 128] = reg1;
    __syncthreads();

    // level 2
    // load
    w_idx = threadIdx.x >> 6;
    butt_idx = (w_idx << 6) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 64];
    // compute
    t = montgomery_multiply(reg1, c_zetas[2 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 64] = reg1;
    __syncthreads();

    // level 3
    // load
    w_idx = threadIdx.x >> 5;
    butt_idx = (w_idx << 5) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 32];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[4 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    butt_idx += ((butt_idx >> 5) << 4);
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 32 + 16] = reg1;
    __syncthreads();

    // level 4
    // load
    w_idx = threadIdx.x >> 4;
    butt_idx = (w_idx << 4) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 5) << 4)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 5) << 4) + 16];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[8 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 4) << 3)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 16 + 8] = reg1;
    __syncthreads();

    // level 5
    // load
    w_idx = threadIdx.x >> 3;
    butt_idx = (w_idx << 3) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 4) << 3)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 8];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[16 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 3) << 2)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 8 + 4] = reg1;
    __syncthreads();

    // level 6
    // load
    w_idx = threadIdx.x >> 2;
    butt_idx = (w_idx << 2) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 3) << 2)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 4];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[32 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 2) << 1)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 4 + 2] = reg1;
    __syncthreads();

    // level 7
    // load
    w_idx = threadIdx.x >> 1;
    butt_idx = (w_idx << 1) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 2) << 1)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 2];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[64 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + (butt_idx >> 1)] = reg0;
    s_ntt[butt_idx + (butt_idx >> 1) + 2 + 1] = reg1;
    __syncthreads();

    // level 8
    // load
    butt_idx = threadIdx.x + threadIdx.x;
    reg0 = s_ntt[butt_idx + (butt_idx >> 1)];
    reg1 = s_ntt[butt_idx + (butt_idx >> 1) + 1];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[128 + threadIdx.x]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 1] = reg1;
    __syncthreads();
}

__device__ void ntt_radix2_inner_opt3(int32_t s_ntt[DILITHIUM_N + 128]) {
    int32_t reg0, reg1;
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // level 1
    // load
    reg0 = s_ntt[threadIdx.x];
    reg1 = s_ntt[threadIdx.x + 128];
    // compute
    t = montgomery_multiply(reg1, c_zetas[1]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[threadIdx.x] = reg0;
    s_ntt[threadIdx.x + 128] = reg1;
    __syncthreads();

    // level 2
    // load
    w_idx = threadIdx.x >> 6;
    butt_idx = (w_idx << 6) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 64];
    // compute
    t = montgomery_multiply(reg1, c_zetas[2 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 64] = reg1;
    __syncthreads();

    // level 3
    // load
    w_idx = threadIdx.x >> 5;
    butt_idx = (w_idx << 5) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 32];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[4 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    butt_idx += ((butt_idx >> 5) << 4);
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 32 + 16] = reg1;

    // level 4
    // load
    w_idx = threadIdx.x >> 4;
    butt_idx = (w_idx << 4) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 5) << 4)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 5) << 4) + 16];
    // compute
    t = montgomery_multiply(reg1, c_zetas[8 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 4) << 3)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 16 + 8] = reg1;

    // level 5
    // load
    w_idx = threadIdx.x >> 3;
    butt_idx = (w_idx << 3) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 4) << 3)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 8];
    // compute
    t = montgomery_multiply(reg1, c_zetas[16 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 3) << 2)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 8 + 4] = reg1;

    // level 6
    // load
    w_idx = threadIdx.x >> 2;
    butt_idx = (w_idx << 2) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 3) << 2)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 4];
    // compute
    t = montgomery_multiply(reg1, c_zetas[32 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 2) << 1)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 4 + 2] = reg1;

    // level 7
    // load
    w_idx = threadIdx.x >> 1;
    butt_idx = (w_idx << 1) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 2) << 1)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 2];
    // compute
    t = montgomery_multiply(reg1, c_zetas[64 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + (butt_idx >> 1)] = reg0;
    s_ntt[butt_idx + (butt_idx >> 1) + 2 + 1] = reg1;

    // level 8
    // load
    butt_idx = threadIdx.x + threadIdx.x;
    reg0 = s_ntt[butt_idx + (butt_idx >> 1)];
    reg1 = s_ntt[butt_idx + (butt_idx >> 1) + 1];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, c_zetas[128 + threadIdx.x]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + (butt_idx >> 5)] = reg0;
    s_ntt[butt_idx + (butt_idx >> 5) + 1] = reg1;
}

__device__ void intt_radix2_inner_opt3(int32_t s_ntt[DILITHIUM_N + 128]) {
    int32_t reg0, reg1;
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // level 8
    butt_idx = threadIdx.x + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + (butt_idx >> 5)];
    reg1 = s_ntt[butt_idx + (butt_idx >> 5) + 1];
    __syncthreads();
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = montgomery_multiply(reg1, -c_zetas[255 - threadIdx.x]);
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
    reg1 = montgomery_multiply(reg1, -c_zetas[127 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -c_zetas[63 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -c_zetas[31 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -c_zetas[15 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -c_zetas[7 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -c_zetas[3 - w_idx]);
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 64] = reg1;
    __syncthreads();

    // level 1
    // load
    reg0 = s_ntt[threadIdx.x];
    reg1 = s_ntt[threadIdx.x + 128];
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = montgomery_multiply(reg1, -c_zetas[1]);
    // divide by N, then store
    s_ntt[threadIdx.x] = montgomery_multiply(reg0, MONT2DIVN);
    s_ntt[threadIdx.x + 128] = montgomery_multiply(reg1, MONT2DIVN);
    __syncthreads();
}

__device__ void ntt_radix2_inner_opt4(int32_t s_ntt[DILITHIUM_N + 128], const int32_t s_zetas[DILITHIUM_N]) {
    int32_t reg0, reg1;
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // level 1
    // load
    reg0 = s_ntt[threadIdx.x];
    reg1 = s_ntt[threadIdx.x + 128];
    // compute
    t = montgomery_multiply(reg1, s_zetas[1]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[threadIdx.x] = reg0;
    s_ntt[threadIdx.x + 128] = reg1;
    __syncthreads();

    // level 2
    // load
    w_idx = threadIdx.x >> 6;
    butt_idx = (w_idx << 6) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 64];
    // compute
    t = montgomery_multiply(reg1, s_zetas[2 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 64] = reg1;
    __syncthreads();

    // level 3
    // load
    w_idx = threadIdx.x >> 5;
    butt_idx = (w_idx << 5) + threadIdx.x;
    reg0 = s_ntt[butt_idx];
    reg1 = s_ntt[butt_idx + 32];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, s_zetas[4 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    butt_idx += ((butt_idx >> 5) << 4);
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 32 + 16] = reg1;

    // level 4
    // load
    w_idx = threadIdx.x >> 4;
    butt_idx = (w_idx << 4) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 5) << 4)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 5) << 4) + 16];
    // compute
    t = montgomery_multiply(reg1, s_zetas[8 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 4) << 3)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 16 + 8] = reg1;

    // level 5
    // load
    w_idx = threadIdx.x >> 3;
    butt_idx = (w_idx << 3) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 4) << 3)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 4) << 3) + 8];
    // compute
    t = montgomery_multiply(reg1, s_zetas[16 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 3) << 2)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 8 + 4] = reg1;

    // level 6
    // load
    w_idx = threadIdx.x >> 2;
    butt_idx = (w_idx << 2) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 3) << 2)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 3) << 2) + 4];
    // compute
    t = montgomery_multiply(reg1, s_zetas[32 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + ((butt_idx >> 2) << 1)] = reg0;
    s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 4 + 2] = reg1;

    // level 7
    // load
    w_idx = threadIdx.x >> 1;
    butt_idx = (w_idx << 1) + threadIdx.x;
    reg0 = s_ntt[butt_idx + ((butt_idx >> 2) << 1)];
    reg1 = s_ntt[butt_idx + ((butt_idx >> 2) << 1) + 2];
    // compute
    t = montgomery_multiply(reg1, s_zetas[64 + w_idx]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + (butt_idx >> 1)] = reg0;
    s_ntt[butt_idx + (butt_idx >> 1) + 2 + 1] = reg1;

    // level 8
    // load
    butt_idx = threadIdx.x + threadIdx.x;
    reg0 = s_ntt[butt_idx + (butt_idx >> 1)];
    reg1 = s_ntt[butt_idx + (butt_idx >> 1) + 1];
    __syncthreads();
    // compute
    t = montgomery_multiply(reg1, s_zetas[128 + threadIdx.x]);
    reg1 = reg0 - t;
    reg0 = reg0 + t;
    // store
    s_ntt[butt_idx + (butt_idx >> 5)] = reg0;
    s_ntt[butt_idx + (butt_idx >> 5) + 1] = reg1;
}

__device__ void intt_radix2_inner_opt4(int32_t s_ntt[DILITHIUM_N + 128], const int32_t s_zetas[DILITHIUM_N]) {
    int32_t reg0, reg1;
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // level 8
    butt_idx = threadIdx.x + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + (butt_idx >> 5)];
    reg1 = s_ntt[butt_idx + (butt_idx >> 5) + 1];
    __syncthreads();
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = montgomery_multiply(reg1, -s_zetas[255 - threadIdx.x]);
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
    reg1 = montgomery_multiply(reg1, -s_zetas[127 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -s_zetas[63 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -s_zetas[31 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -s_zetas[15 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -s_zetas[7 - w_idx]);
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
    reg1 = montgomery_multiply(reg1, -s_zetas[3 - w_idx]);
    // store
    s_ntt[butt_idx] = reg0;
    s_ntt[butt_idx + 64] = reg1;
    __syncthreads();

    // level 1
    // load
    reg0 = s_ntt[threadIdx.x];
    reg1 = s_ntt[threadIdx.x + 128];
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = montgomery_multiply(reg1, -s_zetas[1]);
    // divide by N, then store
    s_ntt[threadIdx.x] = montgomery_multiply(reg0, MONT2DIVN);
    s_ntt[threadIdx.x + 128] = montgomery_multiply(reg1, MONT2DIVN);
    __syncthreads();
}

__device__ void ntt_radix2_inner_opt5(int32_t s_ntt[DILITHIUM_N + 128], const int32_t s_zetas[DILITHIUM_N]) {
    int32_t reg0, reg1;
    size_t butt_idx;
    int32_t t;
    int32_t zeta;

    // level 1
    reg1 = s_ntt[threadIdx.x + 128];
    t = reg1 * 1830766168;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(reg1, 25847) - t;
    reg0 = s_ntt[threadIdx.x];
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
    s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4) + 1] = reg0 - t;
    s_ntt[(threadIdx.x << 1) + (threadIdx.x >> 4)] = reg0 + t;
}

__device__ void intt_radix2_inner_opt5(int32_t s_ntt[DILITHIUM_N + 128], const int32_t s_zetas[DILITHIUM_N]) {
    int32_t reg0, reg1;
    size_t w_idx;
    size_t butt_idx;
    int32_t t;

    // level 8
    butt_idx = threadIdx.x + threadIdx.x;
    // load
    reg0 = s_ntt[butt_idx + (butt_idx >> 5)];
    reg1 = s_ntt[butt_idx + (butt_idx >> 5) + 1];
    __syncthreads();
    // compute
    t = reg0;
    reg0 = t + reg1;
    reg1 = t - reg1;
    reg1 = montgomery_multiply_c(reg1, -s_zetas[255 - threadIdx.x]);
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
    reg1 = montgomery_multiply_c(reg1, -s_zetas[127 - w_idx]);
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
    reg1 = montgomery_multiply_c(reg1, -s_zetas[63 - w_idx]);
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
    reg1 = montgomery_multiply_c(reg1, -s_zetas[31 - w_idx]);
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
    reg1 = montgomery_multiply_c(reg1, -s_zetas[15 - w_idx]);
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
    reg1 = montgomery_multiply_c(reg1, -s_zetas[7 - w_idx]);
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
    reg1 = montgomery_multiply_c(reg1, -s_zetas[3 - w_idx]);
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
    s_ntt[threadIdx.x] = montgomery_multiply_c(reg0, MONT2DIVN);
    reg1 = t - reg1;
    s_ntt[threadIdx.x + 128] = montgomery_multiply_c(reg1, MONT2DIVNMULZETA);
    __syncthreads();
}

__global__ void seo_ntt(int32_t *g_ntt) {
    int32_t *g_per_ntt = g_ntt + gridDim.x * DILITHIUM_N;
    size_t interval_size = DILITHIUM_N / 2;
    for (size_t level = 0; level < 8; level++, interval_size >>= 1) {
        for (size_t thread_id = threadIdx.x; thread_id < DILITHIUM_N / 2; thread_id += 32) {
            size_t section_number = thread_id / interval_size;
            size_t index_number = thread_id % interval_size;
            size_t butt_idx = 2 * thread_id - index_number;
            ntt_butt(g_per_ntt[butt_idx], g_per_ntt[butt_idx + interval_size],
                     c_zetas[(1 << level) + section_number]);
            __syncwarp();
        }
    }
}

__global__ void bench_ntt_radix2_opt0() {
    __shared__ int32_t s_ntt[DILITHIUM_N];
    ntt_radix2_inner_opt0(s_ntt);
}

__global__ void bench_ntt_radix2_opt1() {
    __shared__ int32_t s_ntt[DILITHIUM_N];
    ntt_radix2_inner_opt1(s_ntt);
}

__global__ void bench_ntt_radix2_opt2() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    ntt_radix2_inner_opt2(s_ntt);
}

__global__ void bench_ntt_radix2_opt3() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    ntt_radix2_inner_opt3(s_ntt);
}

__global__ void bench_ntt_radix2_opt4() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];
    ntt_radix2_inner_opt4(s_ntt, s_zetas);
}

__global__ void bench_ntt_radix2_opt5() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];
    ntt_radix2_inner_opt5(s_ntt, s_zetas);
}

__global__ void bench_intt_radix2_opt5() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];
    intt_radix2_inner_opt5(s_ntt, s_zetas);
}

__global__ void test_ntt_radix2_opt0() {
    __shared__ int32_t s_ntt[DILITHIUM_N];

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x) {
        s_ntt[i] = 1;
    }
    __syncthreads();

    ntt_radix2_inner_opt0(s_ntt);
    intt_radix2_inner_opt0(s_ntt);

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x)
        if (s_ntt[i] != MONT)
            printf("test_ntt_radix2_opt0 error\n");
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("test_ntt_radix2_opt0 success\n");
}

__global__ void test_ntt_radix2_opt1() {
    __shared__ int32_t s_ntt[DILITHIUM_N];

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x) {
        s_ntt[i] = 1;
    }
    __syncthreads();

    ntt_radix2_inner_opt1(s_ntt);
    intt_radix2_inner_opt0(s_ntt);

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x)
        if (s_ntt[i] != MONT)
            printf("test_ntt_radix2_opt1 error\n");
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("test_ntt_radix2_opt1 success\n");
}

__global__ void test_ntt_radix2_opt2() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x) {
        s_ntt[i] = 1;
    }
    __syncthreads();

    ntt_radix2_inner_opt2(s_ntt);
    intt_radix2_inner_opt0(s_ntt);

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x)
        if (s_ntt[i] != MONT)
            printf("test_ntt_radix2_opt2 error\n");

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("test_ntt_radix2_opt2 success\n");
}

__global__ void test_ntt_radix2_opt3() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x) {
        s_ntt[i] = 1;
    }
    __syncthreads();

    ntt_radix2_inner_opt3(s_ntt);
    intt_radix2_inner_opt3(s_ntt);

    for (int i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x)
        if (s_ntt[i] != MONT)
            printf("s_ntt[%d] = %d, test_ntt_radix2_opt3 error\n", i, s_ntt[i]);

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("test_ntt_radix2_opt3 success\n");
}

__global__ void test_ntt_radix2_opt4() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x) {
        s_ntt[i] = 1;
        s_zetas[i] = c_zetas[i];
    }
    __syncthreads();

    ntt_radix2_inner_opt4(s_ntt, s_zetas);
    intt_radix2_inner_opt4(s_ntt, s_zetas);

    for (int i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x)
        if (s_ntt[i] != MONT)
            printf("s_ntt[%d] = %d, test_ntt_radix2_opt4 error\n", i, s_ntt[i]);

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("test_ntt_radix2_opt4 success\n");
}

__global__ void test_ntt_radix2_opt5() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];

    for (size_t i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x) {
        s_ntt[i] = 1;
        s_zetas[i] = c_zetas[i];
    }
    __syncthreads();

    ntt_radix2_inner_opt5(s_ntt, s_zetas);
    intt_radix2_inner_opt5(s_ntt, s_zetas);

    for (int i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x)
        if (s_ntt[i] != MONT && s_ntt[i] != DILITHIUM_Q + MONT)
            printf("s_ntt[%d] = %d, test_ntt_radix2_opt5 error\n", i, s_ntt[i]);

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("test_ntt_radix2_opt5 success\n");
}

int main() {
    print_timer_banner();

    for (size_t grid_dim = 10000; grid_dim >= 1; grid_dim /= 10) {
        int32_t *d_ntt;
        cudaMalloc(&d_ntt, sizeof(int32_t) * grid_dim * DILITHIUM_N);

        std::cout << "grid_dim = " << grid_dim << std::endl;
        CUDATimer timer_iopt5("intt radix2 opt 5");
        CUDATimer timer_opt5("ntt radix2 opt 5");
        CUDATimer timer_opt4("ntt radix2 opt 4");
        CUDATimer timer_opt3("ntt radix2 opt 3");
        CUDATimer timer_opt2("ntt radix2 opt 2");
        CUDATimer timer_opt1("ntt radix2 opt 1");
        CUDATimer timer_opt0("ntt radix2 opt 0");
        CUDATimer timer_seo("seo ntt");

        for (size_t i = 0; i < NTESTS; i++) {
            timer_seo.start();
            seo_ntt<<<grid_dim, 32>>>(d_ntt);
            timer_seo.stop();

            timer_opt0.start();
            bench_ntt_radix2_opt0<<<grid_dim, 128>>>();
            timer_opt0.stop();

            timer_opt1.start();
            bench_ntt_radix2_opt1<<<grid_dim, 128>>>();
            timer_opt1.stop();

            timer_opt2.start();
            bench_ntt_radix2_opt2<<<grid_dim, 128>>>();
            timer_opt2.stop();

            timer_opt3.start();
            bench_ntt_radix2_opt3<<<grid_dim, 128>>>();
            timer_opt3.stop();

            timer_opt4.start();
            bench_ntt_radix2_opt4<<<grid_dim, 128>>>();
            timer_opt4.stop();

            timer_opt5.start();
            bench_ntt_radix2_opt5<<<grid_dim, 128>>>();
            timer_opt5.stop();

            timer_iopt5.start();
            bench_intt_radix2_opt5<<<grid_dim, 128>>>();
            timer_iopt5.stop();
        }

        cudaFree(d_ntt);
    }

    CUDATimer timer("test ntt radix2");
    timer.start();
    test_ntt_radix2_opt0<<<10000, 128>>>();
    test_ntt_radix2_opt1<<<10000, 128>>>();
    test_ntt_radix2_opt2<<<10000, 128>>>();
    test_ntt_radix2_opt3<<<10000, 128>>>();
    test_ntt_radix2_opt4<<<10000, 128>>>();
    test_ntt_radix2_opt5<<<10000, 128>>>();
    timer.stop();
    return 0;
}

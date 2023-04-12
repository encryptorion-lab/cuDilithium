#include <cstdint>

#include "keccak.cuh"
#include "ntt.cuh"
#include "params.h"
#include "reduce.cuh"

extern __device__ const int32_t c_zetas[DILITHIUM_N] = {
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

__device__ __inline__ static void ntt_butt(int32_t &a, int32_t &b, const int32_t zeta) {
    int32_t t = gpu_montgomery_multiply(zeta, b);
    b = a - t;
    a = a + t;
}

__device__ __inline__ static void invntt_butt(int32_t &a, int32_t &b, const int32_t zeta) {
    int32_t t = a;
    a = t + b;
    b = t - b;
    b = gpu_montgomery_multiply(zeta, b);
}

/**
 * @brief Forward NTT inner loop
 * @param [out] g_poly output polynomial
 * @param [in] regs pre-allocated registers used for calculating butterflies
 * @param s_poly pre-allocated shared memory used for exchanging coefficients
 */
__device__ void ntt_inner(int32_t regs[8], int32_t s_poly[DILITHIUM_N + 32]) {
    // level 1
    ntt_butt(regs[0], regs[4], c_zetas[1]);
    ntt_butt(regs[1], regs[5], c_zetas[1]);
    ntt_butt(regs[2], regs[6], c_zetas[1]);
    ntt_butt(regs[3], regs[7], c_zetas[1]);
    // level 2
    ntt_butt(regs[0], regs[2], c_zetas[2]);
    ntt_butt(regs[1], regs[3], c_zetas[2]);
    ntt_butt(regs[4], regs[6], c_zetas[3]);
    ntt_butt(regs[5], regs[7], c_zetas[3]);
    // level 3
    ntt_butt(regs[0], regs[1], c_zetas[4]);
    ntt_butt(regs[2], regs[3], c_zetas[5]);
    ntt_butt(regs[4], regs[5], c_zetas[6]);
    ntt_butt(regs[6], regs[7], c_zetas[7]);
    // SMEM exchange
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        s_poly[i * 36 + threadIdx.x] = regs[i];
    __syncwarp();
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        regs[i] = s_poly[(threadIdx.x / 4) * 36 + (threadIdx.x & 3) + i * 4];
    // level 4
    ntt_butt(regs[0], regs[4], c_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[1], regs[5], c_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[2], regs[6], c_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[3], regs[7], c_zetas[8 + threadIdx.x / 4]);
    // level 5
    ntt_butt(regs[0], regs[2], c_zetas[16 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[1], regs[3], c_zetas[16 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[4], regs[6], c_zetas[17 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[5], regs[7], c_zetas[17 + (threadIdx.x / 4) * 2]);
    // level 6
    ntt_butt(regs[0], regs[1], c_zetas[32 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[2], regs[3], c_zetas[33 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[4], regs[5], c_zetas[34 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[6], regs[7], c_zetas[35 + (threadIdx.x / 4) * 4]);
    // SMEM exchange
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        size_t offset_per_row = (threadIdx.x & 3) + i * 4;
        s_poly[(threadIdx.x / 4) * 36 + offset_per_row / 8 + offset_per_row] = regs[i];
    }
    __syncwarp();
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        regs[i] = s_poly[threadIdx.x * 9 + i];
    // level 7
    ntt_butt(regs[0], regs[2], c_zetas[64 + threadIdx.x * 2]);
    ntt_butt(regs[1], regs[3], c_zetas[64 + threadIdx.x * 2]);
    ntt_butt(regs[4], regs[6], c_zetas[65 + threadIdx.x * 2]);
    ntt_butt(regs[5], regs[7], c_zetas[65 + threadIdx.x * 2]);
    // level 8
    ntt_butt(regs[0], regs[1], c_zetas[128 + threadIdx.x * 4]);
    ntt_butt(regs[2], regs[3], c_zetas[129 + threadIdx.x * 4]);
    ntt_butt(regs[4], regs[5], c_zetas[130 + threadIdx.x * 4]);
    ntt_butt(regs[6], regs[7], c_zetas[131 + threadIdx.x * 4]);
}

/**
 * @brief Inverse NTT inner loop
 * @param [inout] regs pre-allocated registers used for calculating butterflies
 * @param s_poly pre-allocated shared memory used for exchanging coefficients
 */
__device__ void invntt_inner(int32_t regs[8], int32_t s_poly[DILITHIUM_N + 32]) {
    // level 1
    invntt_butt(regs[0], regs[1], -c_zetas[256 - threadIdx.x * 4 - 1]);
    invntt_butt(regs[2], regs[3], -c_zetas[256 - threadIdx.x * 4 - 2]);
    invntt_butt(regs[4], regs[5], -c_zetas[256 - threadIdx.x * 4 - 3]);
    invntt_butt(regs[6], regs[7], -c_zetas[256 - threadIdx.x * 4 - 4]);
    // level 2
    invntt_butt(regs[0], regs[2], -c_zetas[128 - threadIdx.x * 2 - 1]);
    invntt_butt(regs[1], regs[3], -c_zetas[128 - threadIdx.x * 2 - 1]);
    invntt_butt(regs[4], regs[6], -c_zetas[128 - threadIdx.x * 2 - 2]);
    invntt_butt(regs[5], regs[7], -c_zetas[128 - threadIdx.x * 2 - 2]);
    // SMEM exchange
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        s_poly[threadIdx.x * 9 + i] = regs[i];
    __syncwarp();
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        size_t offset_per_row = (threadIdx.x & 3) + i * 4;
        regs[i] = s_poly[(threadIdx.x / 4) * 36 + offset_per_row / 8 + offset_per_row];
    }
    // level 3
    invntt_butt(regs[0], regs[1], -c_zetas[64 - (threadIdx.x / 4) * 4 - 1]);
    invntt_butt(regs[2], regs[3], -c_zetas[64 - (threadIdx.x / 4) * 4 - 2]);
    invntt_butt(regs[4], regs[5], -c_zetas[64 - (threadIdx.x / 4) * 4 - 3]);
    invntt_butt(regs[6], regs[7], -c_zetas[64 - (threadIdx.x / 4) * 4 - 4]);
    // level 4
    invntt_butt(regs[0], regs[2], -c_zetas[32 - (threadIdx.x / 4) * 2 - 1]);
    invntt_butt(regs[1], regs[3], -c_zetas[32 - (threadIdx.x / 4) * 2 - 1]);
    invntt_butt(regs[4], regs[6], -c_zetas[32 - (threadIdx.x / 4) * 2 - 2]);
    invntt_butt(regs[5], regs[7], -c_zetas[32 - (threadIdx.x / 4) * 2 - 2]);
    // level 5
    invntt_butt(regs[0], regs[4], -c_zetas[16 - threadIdx.x / 4 - 1]);
    invntt_butt(regs[1], regs[5], -c_zetas[16 - threadIdx.x / 4 - 1]);
    invntt_butt(regs[2], regs[6], -c_zetas[16 - threadIdx.x / 4 - 1]);
    invntt_butt(regs[3], regs[7], -c_zetas[16 - threadIdx.x / 4 - 1]);
    // SMEM exchange
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        s_poly[(threadIdx.x / 4) * 36 + (threadIdx.x & 3) + i * 4] = regs[i];
    __syncwarp();
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        regs[i] = s_poly[i * 36 + threadIdx.x];
    // level 6
    invntt_butt(regs[0], regs[1], -c_zetas[7]);
    invntt_butt(regs[2], regs[3], -c_zetas[6]);
    invntt_butt(regs[4], regs[5], -c_zetas[5]);
    invntt_butt(regs[6], regs[7], -c_zetas[4]);
    // level 7
    invntt_butt(regs[0], regs[2], -c_zetas[3]);
    invntt_butt(regs[1], regs[3], -c_zetas[3]);
    invntt_butt(regs[4], regs[6], -c_zetas[2]);
    invntt_butt(regs[5], regs[7], -c_zetas[2]);
    // level 8
    invntt_butt(regs[0], regs[4], -c_zetas[1]);
    invntt_butt(regs[1], regs[5], -c_zetas[1]);
    invntt_butt(regs[2], regs[6], -c_zetas[1]);
    invntt_butt(regs[3], regs[7], -c_zetas[1]);
    // reduce
    const int32_t f = 41978;// mont^2/256
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        regs[i] = gpu_montgomery_multiply(f, regs[i]);
}

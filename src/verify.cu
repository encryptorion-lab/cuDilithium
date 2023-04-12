#include "verify.cuh"

#include "keccak.cuh"
#include "ntt.cuh"
#include "params.h"
#include "poly.cuh"
#include "reduce.cuh"
#include "rounding.cuh"

#define POLY_UNIFORM_NBLOCKS ((768 + SHAKE128_RATE - 1) / SHAKE128_RATE)

__global__ void gpu_verify(int *g_ret,
                           const uint8_t *g_c, const uint8_t *g_z_packed, const uint8_t *g_h_packed,
                           const uint8_t *g_rho, const uint8_t *g_t1_packed,
                           uint8_t *g_muprime, const uint8_t *g_m, uint8_t *g_mu, uint8_t *g_w1prime_packed, uint8_t *g_ctilde,
                           int32_t *g_mat, int32_t *g_z, int32_t *g_t1, int32_t *g_w1prime, int32_t *g_h, int32_t *g_cp,
                           size_t mlen, size_t mem_pool_pitch) {
    __shared__ uint64_t s_rho64[SEEDBYTES / 8];
    __shared__ uint64_t s_buf_gena64[POLY_UNIFORM_NBLOCKS * SHAKE128_RATE / 8];
    __shared__ uint64_t s_buf64[SHAKE256_RATE / 8];

    __shared__ int32_t s_tmp[DILITHIUM_N + 32];
    __shared__ int32_t s_tmp2[DILITHIUM_N + 32];

    __shared__ int s_fail_flag;

    auto s_rho = (uint8_t *) s_rho64;
    auto s_buf_gena = (uint8_t *) s_buf_gena64;
    auto s_buf = (uint8_t *) s_buf64;

    s_fail_flag = 0;

    int32_t regs[8];

    do {
        // first decode h
        unsigned int pos = 0;
        auto *g_hint_packed = g_h_packed + blockIdx.x * mem_pool_pitch;
        for (size_t k = 0; k < DILITHIUM_K; ++k) {
            auto *g_h_k = g_h + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
            uint8_t sig_omega_k = g_hint_packed[OMEGA + k];
            if (sig_omega_k < pos || sig_omega_k > OMEGA) {
                if (threadIdx.x == 0) s_fail_flag = 1;
                __syncwarp();
                break;
            }
            unsigned int fail_mask = 0;
            for (size_t i = pos + threadIdx.x; i < sig_omega_k; i += 32) {
                // Coefficients are ordered for strong unforgeability
                fail_mask ^= __ballot_sync(__activemask(), i > pos && g_hint_packed[i] <= g_hint_packed[i - 1]);
            }
            __syncwarp();
            fail_mask = __shfl_sync(0xFFFFFFFF, fail_mask, 0);
            if (fail_mask) {
                if (threadIdx.x == 0) s_fail_flag = 1;
                __syncwarp();
                break;
            }
            for (size_t i = pos + threadIdx.x; i < sig_omega_k; i += 32) {
                g_h_k[g_hint_packed[i]] = 1;
            }
            pos = sig_omega_k;
        }
        if (s_fail_flag) break;
        /* Extra indices are zero for strong unforgeability */
        unsigned int fail_mask = 0;
        for (size_t i = pos + threadIdx.x; i < OMEGA; i += 32) {
            // TODO: use match_any warp vote to replace ballot
            fail_mask ^= __ballot_sync(__activemask(), g_hint_packed[i]);
        }
        __syncwarp();
        fail_mask = __shfl_sync(0xFFFFFFFF, fail_mask, 0);
        if (fail_mask) {
            if (threadIdx.x == 0) s_fail_flag = 1;
            __syncwarp();
            break;
        }

        // unpack z to registers in every loop, then check norm and perform NTT
        for (size_t l = 0; l < DILITHIUM_L; ++l) {
            auto *g_z_packed_l = g_z_packed + blockIdx.x * mem_pool_pitch + l * POLYZ_PACKEDBYTES;
            auto *g_z_l = g_z + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + l * DILITHIUM_N;
            for (size_t i = 0; i < 8; i++) {
#if GAMMA1 == (1 << 17)// 18 bit per coefficient
                uint32_t t = (g_z_packed_l[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 0]) |
                             (g_z_packed_l[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 1] << 8) |
                             (g_z_packed_l[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 2] << 16);
                t >>= (threadIdx.x & 3) * 2;
                t &= 0x3FFFF;
#elif GAMMA1 == (1 << 19)// 20 bit per coefficient
                uint32_t t = (g_z_packed_l[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 0]) |
                             (g_z_packed_l[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 1] << 8) |
                             (g_z_packed_l[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 2] << 16);
                t >>= (threadIdx.x & 1) * 4;
                t &= 0xFFFFF;
#endif
                regs[i] = GAMMA1 - (int32_t) t;
                // chknorm
                unsigned int fail_mask = __ballot_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(regs[i]));
                if (fail_mask) {
                    if (threadIdx.x == 0) s_fail_flag = 1;
                    __syncwarp();
                    break;
                }
            }
            if (s_fail_flag) break;
            ntt_inner(regs, s_tmp);
            for (size_t i = 0; i < 8; i++)
                g_z_l[threadIdx.x * 8 + i] = regs[i];
        }
        if (s_fail_flag) break;

        __syncwarp();
        // mu' = H(rho, t1)
        shake<SHAKE256_RATE, 0x1f>(s_buf, 1,
                                   g_rho + blockIdx.x * mem_pool_pitch,
                                   CRYPTO_PUBLICKEYBYTES);
        __syncwarp();
        g_muprime[blockIdx.x * mem_pool_pitch + threadIdx.x] = s_buf[threadIdx.x];
        __syncwarp();
        // mu = CRH(mu', m)
        shake<SHAKE256_RATE, 0x1f>(s_buf, 1,
                                   g_muprime + blockIdx.x * mem_pool_pitch,
                                   SEEDBYTES + mlen);
        __syncwarp();
        g_mu[blockIdx.x * mem_pool_pitch + threadIdx.x] = s_buf[threadIdx.x];
        g_mu[blockIdx.x * mem_pool_pitch + threadIdx.x + 32] = s_buf[threadIdx.x + 32];
        __syncwarp();

        // expand A
        s_rho[threadIdx.x] = g_rho[blockIdx.x * mem_pool_pitch + threadIdx.x];
        __syncwarp();
        for (unsigned int i = 0; i < DILITHIUM_K; ++i) {
            for (unsigned int j = 0; j < DILITHIUM_L; ++j) {
                s_buf_gena[threadIdx.x] = s_rho[threadIdx.x];
                if (threadIdx.x == 0) {
                    s_buf_gena[SEEDBYTES] = (i << 8) + j;
                    s_buf_gena[SEEDBYTES + 1] = i;
                }
                __syncwarp();
                shake<SHAKE128_RATE, 0x1f>(s_buf_gena, POLY_UNIFORM_NBLOCKS, s_buf_gena, SEEDBYTES + 2);
                __syncwarp();

                size_t ctr = 0;
                auto g_poly = g_mat + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + i * DILITHIUM_L * DILITHIUM_N + j * DILITHIUM_N;
                for (size_t round = 0; round < 8; round++) {
                    size_t s_buf_offset = round * 32 * 3 + threadIdx.x * 3;
                    uint8_t t0 = s_buf_gena[s_buf_offset + 0];
                    uint8_t t1 = s_buf_gena[s_buf_offset + 1];
                    uint8_t t2 = s_buf_gena[s_buf_offset + 2] & 0x7F;
                    uint32_t t = t0 | (t1 << 8) | (t2 << 16);

                    int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;

                    unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
                    if (good_mask == 0xFFFFFFFF) {
                        g_poly[ctr + threadIdx.x] = t;
                        ctr += 32;
                    } else {
                        good_mask <<= 31 - threadIdx.x;
                        unsigned int ctr_offset = __popc(good_mask);
                        if (ctr + ctr_offset <= DILITHIUM_N && good) {
                            g_poly[ctr + ctr_offset - 1] = t;
                        }
                        ctr += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
                    }
                }
                if (ctr < DILITHIUM_N) {
                    size_t s_buf_offset = 8 * 32 * 3 + threadIdx.x * 3;
                    uint8_t t0 = s_buf_gena[s_buf_offset + 0];
                    uint8_t t1 = s_buf_gena[s_buf_offset + 1];
                    uint8_t t2 = s_buf_gena[s_buf_offset + 2] & 0x7F;
                    uint32_t t = t0 | (t1 << 8) | (t2 << 16);

                    int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;
                    unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
                    good_mask <<= 31 - threadIdx.x;
                    unsigned int ctr_offset = __popc(good_mask);
                    if (ctr + ctr_offset <= DILITHIUM_N && good) {
                        g_poly[ctr + ctr_offset - 1] = t;
                    }
                }
            }
        }

        // w1' = mat * z
        for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
            auto *g_w1prime_k = g_w1prime + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
            auto *g_mat_k = const_cast<int32_t *>(&g_mat[blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_L * DILITHIUM_N]);
            for (unsigned int i = 0; i < 8; ++i)
                regs[i] = 0;
            for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
                int32_t *g_mat_k_l = g_mat_k + l * DILITHIUM_N;
                int32_t *g_z_l = g_z + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + l * DILITHIUM_N;
                for (unsigned int i = 0; i < 8; ++i)
                    regs[i] += gpu_montgomery_multiply(g_mat_k_l[i * 32 + threadIdx.x], g_z_l[i * 32 + threadIdx.x]);
            }
            for (unsigned int i = 0; i < 8; ++i) {
                g_w1prime_k[32 * i + threadIdx.x] = regs[i];
            }
        }

        // t1 = cp * t1

        // poly_challenge(&cp, c);
        shake<SHAKE256_RATE, 0x1f>(s_buf, 1, g_c + blockIdx.x * mem_pool_pitch, SEEDBYTES);
        __syncwarp();
        if (threadIdx.x == 0) {
            unsigned int i, b, pos;
            uint64_t signs;

            signs = 0;
            for (i = 0; i < 8; ++i)
                signs |= (uint64_t) s_buf[i] << 8 * i;
            pos = 8;

            for (i = 0; i < DILITHIUM_N; ++i)
                s_tmp[i] = 0;
            for (i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
                do {
                    b = s_buf[pos++];
                } while (b > i);

                s_tmp[i] = s_tmp[b];
                s_tmp[b] = 1 - 2 * (signs & 1);
                signs >>= 1;
            }
        }
        __syncwarp();

        // s_tmp = ntt(s_tmp), which stores cp
        for (unsigned int i = 0; i < 8; ++i)
            regs[i] = s_tmp[i * 32 + threadIdx.x];
        ntt_inner(regs, s_tmp);
        for (unsigned int i = 0; i < 8; ++i)
            s_tmp[threadIdx.x * 8 + i] = regs[i];
        __syncwarp();
        // at this time, s_tmp stores ntt(cp), next we need to get t1

        // first we need to unpack t1 from pk and shift left by D bits
        for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
            auto *g_t1_packed_k = g_t1_packed + blockIdx.x * mem_pool_pitch + k * POLYT1_PACKEDBYTES;
            auto *g_w1prime_packed_k = g_w1prime_packed + blockIdx.x * mem_pool_pitch + k * POLYW1_PACKEDBYTES;
            auto *g_w1prime_k = g_w1prime + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
            auto *g_h_k = g_h + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
            for (size_t i = threadIdx.x; i < DILITHIUM_N / 4; i += 32) {
                s_tmp2[4 * i + 0] = (((g_t1_packed_k[5 * i + 0] >> 0) | ((uint32_t) g_t1_packed_k[5 * i + 1] << 8)) & 0x3FF) << DILITHIUM_D;
                s_tmp2[4 * i + 1] = (((g_t1_packed_k[5 * i + 1] >> 2) | ((uint32_t) g_t1_packed_k[5 * i + 2] << 6)) & 0x3FF) << DILITHIUM_D;
                s_tmp2[4 * i + 2] = (((g_t1_packed_k[5 * i + 2] >> 4) | ((uint32_t) g_t1_packed_k[5 * i + 3] << 4)) & 0x3FF) << DILITHIUM_D;
                s_tmp2[4 * i + 3] = (((g_t1_packed_k[5 * i + 3] >> 6) | ((uint32_t) g_t1_packed_k[5 * i + 4] << 2)) & 0x3FF) << DILITHIUM_D;
            }
            __syncwarp();

            for (unsigned int i = 0; i < 8; ++i)
                regs[i] = s_tmp2[i * 32 + threadIdx.x];
            ntt_inner(regs, s_tmp2);
            // at this time, regs stores ntt(t1) in ntt form, then compute t1 = cp * t1
            __syncwarp();

            for (unsigned int i = 0; i < 8; ++i) {
                // s_tmp stores ntt(cp)
                regs[i] = g_w1prime_k[threadIdx.x * 8 + i] - gpu_montgomery_multiply(s_tmp[threadIdx.x * 8 + i], regs[i]);
                // now regs stores w1
                regs[i] = reduce32(regs[i]);
            }
            invntt_inner(regs, s_tmp2);
            for (unsigned int i = 0; i < 8; ++i) {
                // Reconstruct w1, w1 is stored in regs
                regs[i] = caddq(regs[i]);
                s_tmp2[i * 32 + threadIdx.x] = use_hint(regs[i], g_h_k[i * 32 + threadIdx.x]);
            }
            __syncwarp();

            // pack w1 which stored in s_tmp to g_w1prime_packed
#if GAMMA2 == (DILITHIUM_Q - 1) / 88
            for (unsigned int i = 0; i < 2; ++i) {
                g_w1prime_packed_k[6 * threadIdx.x + 3 * i + 0] = (s_tmp2[8 * threadIdx.x + 4 * i + 0]) |
                                                                  (s_tmp2[8 * threadIdx.x + 4 * i + 1] << 6);
                g_w1prime_packed_k[6 * threadIdx.x + 3 * i + 1] = (s_tmp2[8 * threadIdx.x + 4 * i + 1] >> 2) |
                                                                  (s_tmp2[8 * threadIdx.x + 4 * i + 2] << 4);
                g_w1prime_packed_k[6 * threadIdx.x + 3 * i + 2] = (s_tmp2[8 * threadIdx.x + 4 * i + 2] >> 4) |
                                                                  (s_tmp2[8 * threadIdx.x + 4 * i + 3] << 2);
            }
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
            for (unsigned int i = 0; i < 4; ++i)
                g_w1prime_packed_k[4 * threadIdx.x + i] = (s_tmp2[8 * threadIdx.x + 2 * i + 0]) |
                                                          (s_tmp2[8 * threadIdx.x + 2 * i + 1] << 4);
#endif// GAMMA
        }
        __syncwarp();

        // Call random oracle and verify challenge
        shake<SHAKE256_RATE, 0x1f>(s_buf, 1,
                                   g_mu + blockIdx.x * mem_pool_pitch,
                                   CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
        __syncwarp();

        // check c ?= c2
        int not_equal = g_c[blockIdx.x * mem_pool_pitch + threadIdx.x] != s_buf[threadIdx.x];
        if (__ballot_sync(0xFFFFFFFF, not_equal)) {
            if (threadIdx.x == 0) s_fail_flag = 1;
            __syncwarp();
            break;
        }

        if (threadIdx.x == 0) g_ret[blockIdx.x * mem_pool_pitch / sizeof(int)] = 0;

    } while (false);
}

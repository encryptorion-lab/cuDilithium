#include "keypair.cuh"

#include "fips202/fips202.cuh"
#include "ntt.cuh"
#include "params.h"
#include "reduce.cuh"
#include "rounding.cuh"

#define POLY_UNIFORM_NBLOCKS ((768 + SHAKE128_RATE - 1) / SHAKE128_RATE)

#if ETA == 2
#define POLY_UNIFORM_ETA_NBLOCKS ((136 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#elif ETA == 4
#define POLY_UNIFORM_ETA_NBLOCKS ((227 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#endif

__global__ void gpu_keypair(uint8_t *g_pk_rho, uint8_t *g_pk_t1_packed,
                            uint8_t *g_sk_rho, uint8_t *g_sk_key, uint8_t *g_sk_tr,
                            uint8_t *g_sk_s1_packed, uint8_t *g_sk_s2_packed, uint8_t *g_sk_t0_packed,
                            int32_t *g_t1, size_t keypair_mem_pool_pitch, size_t rand_index) {
    __shared__ uint8_t s_random_buf[8];
    __shared__ uint8_t s_seedbuf[SHAKE256_RATE];// 2 * SEEDBYTES + CRHBYTES
    __shared__ uint8_t s_buf_gena[(POLY_UNIFORM_NBLOCKS + 1) * SHAKE128_RATE];
    __shared__ uint8_t s_buf_gens_in[CRHBYTES + 2 + 6];// pad 6 to align to 8 bytes
    __shared__ uint8_t s_buf_gens_out[POLY_UNIFORM_ETA_BUFLEN];

    __shared__ int32_t s_tmp[DILITHIUM_N + 32];
    __shared__ int32_t s_tmp2[DILITHIUM_N + 32];

    int32_t regs[8];

    // get randombytes to s_seedbuf 32 bytes
    // use pseudo random bytes as a workaround to pass testvectors (slightly decrease performance)
    // in real applications, we must use true random bytes
    uint64_t random_ctr = rand_index + blockIdx.x * 2 + 1;
    if (threadIdx.x < 8)
        s_random_buf[threadIdx.x] = random_ctr >> 8 * threadIdx.x;
    __syncwarp();
    shake<SHAKE128_RATE, 0x1F>(s_buf_gens_out, 1, s_random_buf, 8);
    __syncwarp();
    s_seedbuf[threadIdx.x] = s_buf_gens_out[threadIdx.x];
    __syncwarp();

    shake<SHAKE256_RATE, 0x1f>(s_seedbuf, 1, s_seedbuf, SEEDBYTES);
    __syncwarp();

    uint8_t *s_rho = s_seedbuf;
    uint8_t *s_rhoprime = s_rho + SEEDBYTES;
    uint8_t *s_key = s_rhoprime + CRHBYTES;

    // copy s_rho to g_pk_rho
    g_pk_rho[blockIdx.x * keypair_mem_pool_pitch + threadIdx.x] = s_rho[threadIdx.x];
    // copy s_rho to g_sk_rho
    g_sk_rho[blockIdx.x * keypair_mem_pool_pitch + threadIdx.x] = s_rho[threadIdx.x];
    // copy s_key to g_sk_key
    g_sk_key[blockIdx.x * keypair_mem_pool_pitch + threadIdx.x] = s_key[threadIdx.x];
    // copy s_rhoprime to s_buf_gens_in
    s_buf_gens_in[threadIdx.x] = s_rhoprime[threadIdx.x];
    s_buf_gens_in[threadIdx.x + 32] = s_rhoprime[threadIdx.x + 32];
    __syncwarp();

    // expand A
    for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
        // use rhoprime to sample s1 to s_tmp
        // padding two bytes
        if (threadIdx.x == 0) {
            s_buf_gens_in[CRHBYTES + 0] = l & 0xFF;
            s_buf_gens_in[CRHBYTES + 1] = (l >> 8) & 0xFF;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf_gens_out, POLY_UNIFORM_ETA_NBLOCKS, s_buf_gens_in, CRHBYTES + 2);
        __syncwarp();
        unsigned int ctr_rej_eta = 0;
        unsigned int pos_rej_eta = 0;
        while (ctr_rej_eta < DILITHIUM_N && pos_rej_eta + 16 <= POLY_UNIFORM_ETA_BUFLEN) {
            uint8_t buf = s_buf_gens_out[pos_rej_eta + (threadIdx.x >> 1)];
            uint32_t t = (buf >> (4 * (threadIdx.x & 1))) & 0x0F;
#if ETA == 2
            int good = t < 15;
#elif ETA == 4
            int good = t < 9;
#endif
            unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
            good_mask <<= 31 - threadIdx.x;
            unsigned int ctr_offset = __popc(good_mask);
            if (ctr_rej_eta + ctr_offset <= DILITHIUM_N && good) {
#if ETA == 2
                t = t - (205 * t >> 10) * 5;
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 2 - t;
#elif ETA == 4
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 4 - t;
#endif
            }
            ctr_rej_eta += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
            pos_rej_eta += 16;
        }
        __syncwarp();
        // pack s1 into g_sk_s1_packed
        uint8_t *g_sk_s1_packed_l = g_sk_s1_packed + blockIdx.x * keypair_mem_pool_pitch + l * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = 0; i < 8; i++)
            regs[i] = ETA - s_tmp[8 * threadIdx.x + i];
        g_sk_s1_packed_l[3 * threadIdx.x + 0] = (regs[0] >> 0) | (regs[1] << 3) | (regs[2] << 6);
        g_sk_s1_packed_l[3 * threadIdx.x + 1] = (regs[2] >> 2) | (regs[3] << 1) | (regs[4] << 4) | (regs[5] << 7);
        g_sk_s1_packed_l[3 * threadIdx.x + 2] = (regs[5] >> 1) | (regs[6] << 2) | (regs[7] << 5);
#elif ETA == 4
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 2; i += 32) {
            regs[0] = ETA - s_tmp[2 * i + 0];
            regs[1] = ETA - s_tmp[2 * i + 1];
            g_sk_s1_packed_l[i] = regs[0] | (regs[1] << 4);
        }
#endif
        __syncwarp();
        // compute ntt(s1) into s_tmp
        for (size_t i = 0; i < 8; i++)
            regs[i] = s_tmp[i * 32 + threadIdx.x];
        ntt_inner(regs, s_tmp);
        for (size_t i = 0; i < 8; i++)
            s_tmp[8 * threadIdx.x + i] = regs[i];
        __syncwarp();

        for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
            // use g_t1 to accumulate A * s1
            int32_t *g_t1_k = g_t1 + blockIdx.x * keypair_mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
            s_buf_gena[threadIdx.x] = s_rho[threadIdx.x];
            if (threadIdx.x == 0) {
                s_buf_gena[SEEDBYTES] = (k << 8) + l;
                s_buf_gena[SEEDBYTES + 1] = k;
            }
            __syncwarp();
            shake<SHAKE128_RATE, 0x1f>(s_buf_gena, POLY_UNIFORM_NBLOCKS, s_buf_gena, SEEDBYTES + 2);
            __syncwarp();

            size_t ctr = 0;
            for (size_t round = 0; round < 8; round++) {
                size_t s_buf_offset = round * 32 * 3 + threadIdx.x * 3;
                uint8_t t0 = s_buf_gena[s_buf_offset + 0];
                uint8_t t1 = s_buf_gena[s_buf_offset + 1];
                uint8_t t2 = s_buf_gena[s_buf_offset + 2] & 0x7F;
                uint32_t t = t0 | (t1 << 8) | (t2 << 16);

                int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;

                unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
                if (good_mask == 0xFFFFFFFF) {
                    s_tmp2[ctr + threadIdx.x] = t;
                    ctr += 32;
                } else {
                    good_mask <<= 31 - threadIdx.x;
                    unsigned int ctr_offset = __popc(good_mask);
                    if (ctr + ctr_offset <= DILITHIUM_N && good) {
                        s_tmp2[ctr + ctr_offset - 1] = t;
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
                    s_tmp2[ctr + ctr_offset - 1] = t;
                }
            }
            __syncwarp();
            for (size_t i = 0; i < 8; ++i) {
                g_t1_k[32 * i + threadIdx.x] += gpu_montgomery_multiply(s_tmp2[i * 32 + threadIdx.x], s_tmp[i * 32 + threadIdx.x]);
            }
            __syncwarp();
        }
        __syncwarp();
    }

    for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
        // use rhoprime to sample s2 to s_tmp
        // padding two bytes
        if (threadIdx.x == 0) {
            s_buf_gens_in[CRHBYTES + 0] = (DILITHIUM_L + k) & 0xFF;
            s_buf_gens_in[CRHBYTES + 1] = (DILITHIUM_L + k) >> 8;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf_gens_out, POLY_UNIFORM_ETA_NBLOCKS, s_buf_gens_in, CRHBYTES + 2);
        __syncwarp();
        unsigned int ctr_rej_eta = 0;
        unsigned int pos_rej_eta = 0;
        while (ctr_rej_eta < DILITHIUM_N && pos_rej_eta + 16 <= POLY_UNIFORM_ETA_BUFLEN) {
            uint8_t buf = s_buf_gens_out[pos_rej_eta + (threadIdx.x >> 1)];
            uint32_t t = (buf >> (4 * (threadIdx.x & 1))) & 0x0F;
#if ETA == 2
            int good = t < 15;
#elif ETA == 4
            int good = t < 9;
#endif
            unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
            good_mask <<= 31 - threadIdx.x;
            unsigned int ctr_offset = __popc(good_mask);
            if (ctr_rej_eta + ctr_offset <= DILITHIUM_N && good) {
#if ETA == 2
                t = t - (205 * t >> 10) * 5;
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 2 - t;
#elif ETA == 4
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 4 - t;
#endif
            }
            ctr_rej_eta += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
            pos_rej_eta += 16;
        }
        __syncwarp();
        // pack s2 into g_sk_s2_packed
        uint8_t *g_sk_s2_packed_k = g_sk_s2_packed + blockIdx.x * keypair_mem_pool_pitch + k * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = 0; i < 8; i++)
            regs[i] = ETA - s_tmp[8 * threadIdx.x + i];
        g_sk_s2_packed_k[3 * threadIdx.x + 0] = (regs[0] >> 0) | (regs[1] << 3) | (regs[2] << 6);
        g_sk_s2_packed_k[3 * threadIdx.x + 1] = (regs[2] >> 2) | (regs[3] << 1) | (regs[4] << 4) | (regs[5] << 7);
        g_sk_s2_packed_k[3 * threadIdx.x + 2] = (regs[5] >> 1) | (regs[6] << 2) | (regs[7] << 5);
#elif ETA == 4
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 2; i += 32) {
            regs[0] = ETA - s_tmp[2 * i + 0];
            regs[1] = ETA - s_tmp[2 * i + 1];
            g_sk_s2_packed_k[i] = regs[0] | (regs[1] << 4);
        }
#endif
        __syncwarp();

        int32_t *g_t1_k = g_t1 + blockIdx.x * keypair_mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        for (size_t i = 0; i < 8; ++i) {
            regs[i] = g_t1_k[8 * threadIdx.x + i];
            regs[i] = reduce32(regs[i]);
        }
        invntt_inner(regs, s_tmp2);

        for (size_t i = 0; i < 8; ++i) {
            // add s2 to t1
            regs[i] += s_tmp[i * 32 + threadIdx.x];
            regs[i] = caddq(regs[i]);
            s_tmp[i * 32 + threadIdx.x] = power2round(&s_tmp2[i * 32 + threadIdx.x], regs[i]);
            // after that, s_tmp stores t1, s_tmp2 stores t0
        }
        __syncwarp();

        // pack t1 into g_pk_t1_packed
        uint8_t *g_sk_t1_packed_k = g_pk_t1_packed + blockIdx.x * keypair_mem_pool_pitch + k * POLYT1_PACKEDBYTES;
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 4; i += 32) {
            g_sk_t1_packed_k[5 * i + 0] = (s_tmp[4 * i + 0] >> 0);
            g_sk_t1_packed_k[5 * i + 1] = (s_tmp[4 * i + 0] >> 8) | (s_tmp[4 * i + 1] << 2);
            g_sk_t1_packed_k[5 * i + 2] = (s_tmp[4 * i + 1] >> 6) | (s_tmp[4 * i + 2] << 4);
            g_sk_t1_packed_k[5 * i + 3] = (s_tmp[4 * i + 2] >> 4) | (s_tmp[4 * i + 3] << 6);
            g_sk_t1_packed_k[5 * i + 4] = (s_tmp[4 * i + 3] >> 2);
        }
        __syncwarp();

        // pack t0 into g_sk_t0_packed
        uint8_t *g_sk_t0_packed_k = g_sk_t0_packed + blockIdx.x * keypair_mem_pool_pitch + k * POLYT0_PACKEDBYTES;
        for (size_t i = 0; i < 8; ++i)
            regs[i] = (1 << (DILITHIUM_D - 1)) - s_tmp2[8 * threadIdx.x + i];
        g_sk_t0_packed_k[13 * threadIdx.x + 0] = regs[0];
        g_sk_t0_packed_k[13 * threadIdx.x + 1] = regs[0] >> 8;
        g_sk_t0_packed_k[13 * threadIdx.x + 1] |= regs[1] << 5;
        g_sk_t0_packed_k[13 * threadIdx.x + 2] = regs[1] >> 3;
        g_sk_t0_packed_k[13 * threadIdx.x + 3] = regs[1] >> 11;
        g_sk_t0_packed_k[13 * threadIdx.x + 3] |= regs[2] << 2;
        g_sk_t0_packed_k[13 * threadIdx.x + 4] = regs[2] >> 6;
        g_sk_t0_packed_k[13 * threadIdx.x + 4] |= regs[3] << 7;
        g_sk_t0_packed_k[13 * threadIdx.x + 5] = regs[3] >> 1;
        g_sk_t0_packed_k[13 * threadIdx.x + 6] = regs[3] >> 9;
        g_sk_t0_packed_k[13 * threadIdx.x + 6] |= regs[4] << 4;
        g_sk_t0_packed_k[13 * threadIdx.x + 7] = regs[4] >> 4;
        g_sk_t0_packed_k[13 * threadIdx.x + 8] = regs[4] >> 12;
        g_sk_t0_packed_k[13 * threadIdx.x + 8] |= regs[5] << 1;
        g_sk_t0_packed_k[13 * threadIdx.x + 9] = regs[5] >> 7;
        g_sk_t0_packed_k[13 * threadIdx.x + 9] |= regs[6] << 6;
        g_sk_t0_packed_k[13 * threadIdx.x + 10] = regs[6] >> 2;
        g_sk_t0_packed_k[13 * threadIdx.x + 11] = regs[6] >> 10;
        g_sk_t0_packed_k[13 * threadIdx.x + 11] |= regs[7] << 3;
        g_sk_t0_packed_k[13 * threadIdx.x + 12] = regs[7] >> 5;
        __syncwarp();
    }

    // tr = H(rho, t1)
    // reuse s_buf_gens_out
    shake<SHAKE256_RATE, 0x1f>(s_buf_gens_out, 1, g_pk_rho + blockIdx.x * keypair_mem_pool_pitch, CRYPTO_PUBLICKEYBYTES);
    __syncwarp();
    // write tr to g_sk_tr
    g_sk_tr[blockIdx.x * keypair_mem_pool_pitch + threadIdx.x] = s_buf_gens_out[threadIdx.x];
}

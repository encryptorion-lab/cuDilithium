#include "sign.cuh"

#include "fips202/fips202.cuh"
#include "ntt.cuh"
#include "params.h"
#include "poly.cuh"
#include "reduce.cuh"
#include "rounding.cuh"

#define POLY_UNIFORM_NBLOCKS ((768 + SHAKE128_RATE - 1) / SHAKE128_RATE)

#define POLY_UNIFORM_GAMMA1_NBLOCKS ((POLYZ_PACKEDBYTES + SHAKE256_RATE - 1) / SHAKE256_RATE)

#if ETA == 2
#define POLY_UNIFORM_ETA_NBLOCKS ((136 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#elif ETA == 4
#define POLY_UNIFORM_ETA_NBLOCKS ((227 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#endif

__global__ void polyvec_matrix_expand_kernel(int32_t *g_mat, const uint8_t *g_rho, size_t mem_pool_pitch) {
    __shared__ uint8_t s_rho[SEEDBYTES];
    __shared__ uint8_t s_buf[POLY_UNIFORM_NBLOCKS * SHAKE128_RATE];

    s_rho[threadIdx.x] = g_rho[blockIdx.x * mem_pool_pitch + threadIdx.x];

    for (unsigned int i = 0; i < DILITHIUM_K; ++i) {
        for (unsigned int j = 0; j < DILITHIUM_L; ++j) {
            s_buf[threadIdx.x] = s_rho[threadIdx.x];
            if (threadIdx.x == 0) {
                s_buf[SEEDBYTES] = (i << 8) + j;
                s_buf[SEEDBYTES + 1] = i;
            }
            __syncwarp();
            shake<SHAKE128_RATE, 0x1f>(s_buf, POLY_UNIFORM_NBLOCKS, s_buf, SEEDBYTES + 2);
            __syncwarp();

            size_t ctr = 0;
            auto g_poly = g_mat + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + i * DILITHIUM_L * DILITHIUM_N + j * DILITHIUM_N;
            for (size_t round = 0; round < 8; round++) {
                size_t s_buf_offset = round * 32 * 3 + threadIdx.x * 3;
                uint8_t t0 = s_buf[s_buf_offset + 0];
                uint8_t t1 = s_buf[s_buf_offset + 1];
                uint8_t t2 = s_buf[s_buf_offset + 2] & 0x7F;
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
                uint8_t t0 = s_buf[s_buf_offset + 0];
                uint8_t t1 = s_buf[s_buf_offset + 1];
                uint8_t t2 = s_buf[s_buf_offset + 2] & 0x7F;
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
}

__global__ void polyvec_matrix_expand_opt_kernel(int32_t *g_mat, const uint8_t *g_rho, size_t mem_pool_pitch, size_t n_inputs) {
    __shared__ uint8_t s_rho[4][SEEDBYTES];
    __shared__ uint8_t s_buf[4][POLY_UNIFORM_NBLOCKS * SHAKE128_RATE];

    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (input_id < n_inputs) {

        s_rho[threadIdx.y][threadIdx.x] = g_rho[input_id * mem_pool_pitch + threadIdx.x];

        uint8_t *s_buf_y = s_buf[threadIdx.y];

        for (unsigned int i = 0; i < DILITHIUM_K; ++i) {
            for (unsigned int j = 0; j < DILITHIUM_L; ++j) {
                s_buf_y[threadIdx.x] = s_rho[threadIdx.y][threadIdx.x];
                if (threadIdx.x == 0) {
                    s_buf_y[SEEDBYTES] = (i << 8) + j;
                    s_buf_y[SEEDBYTES + 1] = i;
                }
                __syncwarp();
                shake<SHAKE128_RATE, 0x1f>(s_buf_y, POLY_UNIFORM_NBLOCKS, s_buf_y, SEEDBYTES + 2);
                __syncwarp();

                size_t ctr = 0;
                auto g_poly = g_mat + input_id * mem_pool_pitch / sizeof(int32_t) + i * DILITHIUM_L * DILITHIUM_N + j * DILITHIUM_N;
                for (size_t round = 0; round < 8; round++) {
                    size_t s_buf_offset = round * 32 * 3 + threadIdx.x * 3;
                    uint8_t t0 = s_buf_y[s_buf_offset + 0];
                    uint8_t t1 = s_buf_y[s_buf_offset + 1];
                    uint8_t t2 = s_buf_y[s_buf_offset + 2] & 0x7F;
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
                    uint8_t t0 = s_buf_y[s_buf_offset + 0];
                    uint8_t t1 = s_buf_y[s_buf_offset + 1];
                    uint8_t t2 = s_buf_y[s_buf_offset + 2] & 0x7F;
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
    }
}

template<unsigned int VEC_SIZE, unsigned int PACKED_BYTES>
__global__ void unpack_fuse_ntt_kernel(int32_t *g_polyvec, const uint8_t *g_polyvec_packed, size_t mem_pool_pitch) {
    __shared__ int32_t s_poly[DILITHIUM_N + 32];
    int32_t regs[8];

    for (int vec_i = 0; vec_i < VEC_SIZE; vec_i++) {
        const uint8_t *g_poly_packed = g_polyvec_packed + blockIdx.x * mem_pool_pitch + vec_i * PACKED_BYTES;
        for (size_t i = 0; i < 8; i++) {
            if (PACKED_BYTES == POLYETA_PACKEDBYTES) {
#if ETA == 2
                uint32_t t = (g_poly_packed[i * 12 + (threadIdx.x / 8) * 3 + 0]) |
                             (g_poly_packed[i * 12 + (threadIdx.x / 8) * 3 + 1] << 8) |
                             (g_poly_packed[i * 12 + (threadIdx.x / 8) * 3 + 2] << 16);
                t >>= (threadIdx.x & 7) * 3;
                t &= 0x7;
                regs[i] = ETA - (int32_t) t;
#elif ETA == 4
                uint32_t t = g_poly_packed[i * 16 + threadIdx.x / 2];
                t >>= (threadIdx.x & 1) * 4;
                t &= 0xF;
                regs[i] = ETA - (int32_t) t;
#endif// ETA
            } else if (PACKED_BYTES == POLYT0_PACKEDBYTES) {
                uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 0]) |
                             (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 1] << 8) |
                             (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 2] << 16) |
                             (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 3] << 24);
                t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
                t &= 0x1FFF;
                regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t) t;
            }
        }

        int32_t *g_poly = g_polyvec + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + vec_i * DILITHIUM_N;
        ntt_inner(regs, s_poly);
        for (size_t i = 0; i < 8; i++)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

template __global__ void unpack_fuse_ntt_kernel<DILITHIUM_L, POLYETA_PACKEDBYTES>(int32_t *g_polyvec, const uint8_t *g_polyvec_packed, size_t mem_pool_pitch);
#if DILITHIUM_L != DILITHIUM_K
template __global__ void unpack_fuse_ntt_kernel<DILITHIUM_K, POLYETA_PACKEDBYTES>(int32_t *g_polyvec, const uint8_t *g_polyvec_packed, size_t mem_pool_pitch);
#endif
template __global__ void unpack_fuse_ntt_kernel<DILITHIUM_K, POLYT0_PACKEDBYTES>(int32_t *g_polyvec, const uint8_t *g_polyvec_packed, size_t mem_pool_pitch);

__global__ void unpack_fuse_ntt_radix2_kernel(
        int32_t *g_s1, int32_t *g_s2, int32_t *g_t0,
        const uint8_t *g_s1_packed, const uint8_t *g_s2_packed, const uint8_t *g_t0_packed,
        size_t mem_pool_pitch) {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];

    s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
    s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
    __syncthreads();

    for (size_t l = 0; l < DILITHIUM_L; l++) {
        const uint8_t *a = g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
            s_ntt[8 * i + 0] = a[3 * i + 0] >> 0;
            s_ntt[8 * i + 1] = a[3 * i + 0] >> 3;
            s_ntt[8 * i + 2] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
            s_ntt[8 * i + 3] = a[3 * i + 1] >> 1;
            s_ntt[8 * i + 4] = a[3 * i + 1] >> 4;
            s_ntt[8 * i + 5] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
            s_ntt[8 * i + 6] = a[3 * i + 2] >> 2;
            s_ntt[8 * i + 7] = a[3 * i + 2] >> 5;
        }
        __syncthreads();
        int32_t reg0 = ETA - (s_ntt[threadIdx.x] & 7);
        int32_t reg1 = ETA - (s_ntt[128 + threadIdx.x] & 7);
#elif ETA == 4
        uint8_t t = a[threadIdx.x];
        s_ntt[2 * threadIdx.x + 0] = t & 0x0F;
        s_ntt[2 * threadIdx.x + 1] = t >> 4;
        __syncthreads();
        int32_t reg0 = ETA - s_ntt[threadIdx.x];
        int32_t reg1 = ETA - s_ntt[128 + threadIdx.x];
#endif
        ntt_radix2_inner(reg0, reg1, s_ntt, s_zetas);
        int32_t *g_s1_l = g_s1 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + l * DILITHIUM_N;
        g_s1_l[threadIdx.x] = reg0;
        g_s1_l[128 + threadIdx.x] = reg1;
        __syncthreads();
    }

    for (size_t k = 0; k < DILITHIUM_K; k++) {
        const uint8_t *a = g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
            s_ntt[8 * i + 0] = a[3 * i + 0] >> 0;
            s_ntt[8 * i + 1] = a[3 * i + 0] >> 3;
            s_ntt[8 * i + 2] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
            s_ntt[8 * i + 3] = a[3 * i + 1] >> 1;
            s_ntt[8 * i + 4] = a[3 * i + 1] >> 4;
            s_ntt[8 * i + 5] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
            s_ntt[8 * i + 6] = a[3 * i + 2] >> 2;
            s_ntt[8 * i + 7] = a[3 * i + 2] >> 5;
        }
        __syncthreads();
        int32_t reg0 = ETA - (s_ntt[threadIdx.x] & 7);
        int32_t reg1 = ETA - (s_ntt[128 + threadIdx.x] & 7);
#elif ETA == 4
        s_ntt[2 * threadIdx.x + 0] = a[threadIdx.x] & 0x0F;
        s_ntt[2 * threadIdx.x + 1] = a[threadIdx.x] >> 4;
        __syncthreads();
        int32_t reg0 = ETA - s_ntt[threadIdx.x];
        int32_t reg1 = ETA - s_ntt[128 + threadIdx.x];
#endif
        ntt_radix2_inner(reg0, reg1, s_ntt, s_zetas);
        int32_t *g_s2_k = g_s2 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        g_s2_k[threadIdx.x] = reg0;
        g_s2_k[128 + threadIdx.x] = reg1;
        __syncthreads();
    }

    for (size_t k = 0; k < DILITHIUM_K; k++) {
        const uint8_t *a = g_t0_packed + blockIdx.x * mem_pool_pitch + k * POLYT0_PACKEDBYTES;
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
            s_ntt[8 * i + 0] = a[13 * i + 0] | (a[13 * i + 1] << 8);
            s_ntt[8 * i + 1] = (a[13 * i + 1] >> 5) | (a[13 * i + 2] << 3) | (a[13 * i + 3] << 11);
            s_ntt[8 * i + 2] = (a[13 * i + 3] >> 2) | (a[13 * i + 4] << 6);
            s_ntt[8 * i + 3] = (a[13 * i + 4] >> 7) | (a[13 * i + 5] << 1) | (a[13 * i + 6] << 9);
            s_ntt[8 * i + 4] = (a[13 * i + 6] >> 4) | (a[13 * i + 7] << 4) | (a[13 * i + 8] << 12);
            s_ntt[8 * i + 5] = (a[13 * i + 8] >> 1) | (a[13 * i + 9] << 7);
            s_ntt[8 * i + 6] = (a[13 * i + 9] >> 6) | (a[13 * i + 10] << 2) | (a[13 * i + 11] << 10);
            s_ntt[8 * i + 7] = (a[13 * i + 11] >> 3) | (a[13 * i + 12] << 5);
        }
        __syncthreads();
        int32_t reg0 = (1 << (DILITHIUM_D - 1)) - (s_ntt[threadIdx.x] & 0x1FFF);
        int32_t reg1 = (1 << (DILITHIUM_D - 1)) - (s_ntt[128 + threadIdx.x] & 0x1FFF);
        ntt_radix2_inner(reg0, reg1, s_ntt, s_zetas);
        int32_t *g_t0_k = g_t0 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        g_t0_k[threadIdx.x] = reg0;
        g_t0_k[128 + threadIdx.x] = reg1;
        __syncthreads();
    }
}

__global__ void unpack_fuse_ntt_radix2_opt_kernel(
        int32_t *g_s1, int32_t *g_s2, int32_t *g_t0,
        const uint8_t *g_s1_packed, const uint8_t *g_s2_packed, const uint8_t *g_t0_packed,
        size_t mem_pool_pitch) {
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];

    s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
    s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
    __syncthreads();

    for (size_t l = 0; l < DILITHIUM_L; l++) {
#if ETA == 2
        const uint8_t *a = g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
            s_ntt[8 * i + 0 + (i >> 2)] = a[3 * i + 0] >> 0;
            s_ntt[8 * i + 1 + (i >> 2)] = a[3 * i + 0] >> 3;
            s_ntt[8 * i + 2 + (i >> 2)] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
            s_ntt[8 * i + 3 + (i >> 2)] = a[3 * i + 1] >> 1;
            s_ntt[8 * i + 4 + (i >> 2)] = a[3 * i + 1] >> 4;
            s_ntt[8 * i + 5 + (i >> 2)] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
            s_ntt[8 * i + 6 + (i >> 2)] = a[3 * i + 2] >> 2;
            s_ntt[8 * i + 7 + (i >> 2)] = a[3 * i + 2] >> 5;
        }
        __syncthreads();
        int32_t reg0 = ETA - (s_ntt[threadIdx.x + (threadIdx.x >> 5)] & 7);
        int32_t reg1 = ETA - (s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] & 7);
#elif ETA == 4
        const uint8_t *g_s1_packed_l = g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
        uint8_t t = g_s1_packed_l[threadIdx.x];
        s_ntt[2 * threadIdx.x + 0 + (threadIdx.x >> 4)] = t & 0x0F;
        s_ntt[2 * threadIdx.x + 1 + (threadIdx.x >> 4)] = t >> 4;
        __syncthreads();
        int32_t reg0 = ETA - s_ntt[threadIdx.x + (threadIdx.x >> 5)];
        int32_t reg1 = ETA - s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)];
#endif
        __syncthreads();
        ntt_radix2_inner_pad1(reg0, reg1, s_ntt, s_zetas);
        int32_t *g_s1_l = g_s1 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + l * DILITHIUM_N;
        g_s1_l[threadIdx.x] = reg0;
        g_s1_l[128 + threadIdx.x] = reg1;
        __syncthreads();
    }

    for (size_t k = 0; k < DILITHIUM_K; k++) {
#if ETA == 2
        const uint8_t *a = g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
            s_ntt[8 * i + 0 + (i >> 2)] = a[3 * i + 0] >> 0;
            s_ntt[8 * i + 1 + (i >> 2)] = a[3 * i + 0] >> 3;
            s_ntt[8 * i + 2 + (i >> 2)] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
            s_ntt[8 * i + 3 + (i >> 2)] = a[3 * i + 1] >> 1;
            s_ntt[8 * i + 4 + (i >> 2)] = a[3 * i + 1] >> 4;
            s_ntt[8 * i + 5 + (i >> 2)] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
            s_ntt[8 * i + 6 + (i >> 2)] = a[3 * i + 2] >> 2;
            s_ntt[8 * i + 7 + (i >> 2)] = a[3 * i + 2] >> 5;
        }
        __syncthreads();
        int32_t reg0 = ETA - (s_ntt[threadIdx.x + (threadIdx.x >> 5)] & 7);
        int32_t reg1 = ETA - (s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] & 7);
#elif ETA == 4
        const uint8_t *g_s2_packed_k = g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
        uint8_t t = g_s2_packed_k[threadIdx.x];
        s_ntt[2 * threadIdx.x + 0 + (threadIdx.x >> 4)] = t & 0x0F;
        s_ntt[2 * threadIdx.x + 1 + (threadIdx.x >> 4)] = t >> 4;
        __syncthreads();
        int32_t reg0 = ETA - s_ntt[threadIdx.x + (threadIdx.x >> 5)];
        int32_t reg1 = ETA - s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)];
#endif
        ntt_radix2_inner_pad1(reg0, reg1, s_ntt, s_zetas);
        int32_t *g_s2_k = g_s2 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        g_s2_k[threadIdx.x] = reg0;
        g_s2_k[128 + threadIdx.x] = reg1;
        __syncthreads();
    }

    for (size_t k = 0; k < DILITHIUM_K; k++) {
        const uint8_t *a = g_t0_packed + blockIdx.x * mem_pool_pitch + k * POLYT0_PACKEDBYTES;
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
            s_ntt[8 * i + 0 + (i >> 2)] = a[13 * i + 0] | (a[13 * i + 1] << 8);
            s_ntt[8 * i + 1 + (i >> 2)] = (a[13 * i + 1] >> 5) | (a[13 * i + 2] << 3) | (a[13 * i + 3] << 11);
            s_ntt[8 * i + 2 + (i >> 2)] = (a[13 * i + 3] >> 2) | (a[13 * i + 4] << 6);
            s_ntt[8 * i + 3 + (i >> 2)] = (a[13 * i + 4] >> 7) | (a[13 * i + 5] << 1) | (a[13 * i + 6] << 9);
            s_ntt[8 * i + 4 + (i >> 2)] = (a[13 * i + 6] >> 4) | (a[13 * i + 7] << 4) | (a[13 * i + 8] << 12);
            s_ntt[8 * i + 5 + (i >> 2)] = (a[13 * i + 8] >> 1) | (a[13 * i + 9] << 7);
            s_ntt[8 * i + 6 + (i >> 2)] = (a[13 * i + 9] >> 6) | (a[13 * i + 10] << 2) | (a[13 * i + 11] << 10);
            s_ntt[8 * i + 7 + (i >> 2)] = (a[13 * i + 11] >> 3) | (a[13 * i + 12] << 5);
        }
        __syncthreads();
        int32_t reg0 = (1 << (DILITHIUM_D - 1)) - (s_ntt[threadIdx.x + (threadIdx.x >> 5)] & 0x1FFF);
        int32_t reg1 = (1 << (DILITHIUM_D - 1)) - (s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] & 0x1FFF);
        ntt_radix2_inner_pad1(reg0, reg1, s_ntt, s_zetas);
        int32_t *g_t0_k = g_t0 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        g_t0_k[threadIdx.x] = reg0;
        g_t0_k[128 + threadIdx.x] = reg1;
        __syncthreads();
    }
}

// shake, only fuse with unpack
__global__ void compute_y_kernel(int32_t *g_y,
                                 const uint8_t *g_rhoprime,
                                 const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
    __shared__ uint8_t s_in_buf[CRHBYTES + 2 + 6];// pad 6 to align 8 bytes
    __shared__ uint8_t s_buf[POLY_UNIFORM_GAMMA1_NBLOCKS * SHAKE256_RATE];

    uint32_t exec_info = g_exec_lut[blockIdx.x];
    uint16_t nonce = exec_info & 0xFFFF;

    size_t sign_idx_offset = (exec_info >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                     ? sign_idx_offset
                                     : blockIdx.x * temp_mem_pool_pitch;

    // copy rhoprime to shared memory for expand mask
    s_in_buf[threadIdx.x] = g_rhoprime[sign_idx_offset + threadIdx.x];
    s_in_buf[threadIdx.x + 32] = g_rhoprime[sign_idx_offset + threadIdx.x + 32];

    for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
        // padding two bytes
        if (threadIdx.x == 0) {
            // write kappa to shared memory for expand mask
            uint16_t nonce_l = DILITHIUM_L * nonce + l;
            s_in_buf[CRHBYTES + 0] = nonce_l & 0xFF;
            s_in_buf[CRHBYTES + 1] = nonce_l >> 8;
        }
        __syncwarp();
        // y = expand_mask(rhoprime || kappa)
        shake<SHAKE256_RATE, 0x1f>(s_buf, POLY_UNIFORM_GAMMA1_NBLOCKS, s_in_buf, CRHBYTES + 2);
        __syncwarp();

        // unpack y
        int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        // 18 bit per coefficient
#if GAMMA1 == (1 << 17)
        for (size_t i = 0; i < 8; i++) {
            uint32_t t = (s_buf[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 0]) |
                         (s_buf[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 1] << 8) |
                         (s_buf[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 2] << 16);
            t >>= (threadIdx.x & 3) * 2;
            t &= 0x3FFFF;
            g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t) t;
        }
        // 20 bit per coefficient
#elif GAMMA1 == (1 << 19)
        for (size_t i = 0; i < 8; i++) {
            uint32_t t = (s_buf[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 0]) |
                         (s_buf[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 1] << 8) |
                         (s_buf[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 2] << 16);
            t >>= (threadIdx.x & 1) * 4;
            t &= 0xFFFFF;
            g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t) t;
        }
#endif
    }
}

__global__ void compute_y_opt_kernel(int32_t *g_y,
                                     const uint8_t *g_rhoprime,
                                     const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch, size_t n_inputs) {
    __shared__ uint8_t s_in_buf[4][CRHBYTES + 2 + 6];// pad 6 to align 8 bytes
    __shared__ uint8_t s_buf[4][POLY_UNIFORM_GAMMA1_NBLOCKS * SHAKE256_RATE];

    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (input_id < n_inputs) {

        uint8_t *s_in_buf_y = s_in_buf[threadIdx.y];
        uint8_t *s_buf_y = s_buf[threadIdx.y];

        uint32_t exec_info = g_exec_lut[input_id];
        uint16_t nonce = exec_info & 0xFFFF;

        size_t sign_idx_offset = (exec_info >> 16) * sign_mem_pool_pitch;
        size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                         ? sign_idx_offset
                                         : input_id * temp_mem_pool_pitch;

        // copy rhoprime to shared memory for expand mask
        s_in_buf_y[threadIdx.x] = g_rhoprime[sign_idx_offset + threadIdx.x];
        s_in_buf_y[threadIdx.x + 32] = g_rhoprime[sign_idx_offset + threadIdx.x + 32];

        for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
            // padding two bytes
            if (threadIdx.x == 0) {
                // write kappa to shared memory for expand mask
                uint16_t nonce_l = DILITHIUM_L * nonce + l;
                s_in_buf_y[CRHBYTES + 0] = nonce_l & 0xFF;
                s_in_buf_y[CRHBYTES + 1] = nonce_l >> 8;
            }
            __syncwarp();
            // y = expand_mask(rhoprime || kappa)
            shake<SHAKE256_RATE, 0x1f>(s_buf_y, POLY_UNIFORM_GAMMA1_NBLOCKS, s_in_buf_y, CRHBYTES + 2);
            __syncwarp();

            // unpack y
            int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
            // 18 bit per coefficient
#if GAMMA1 == (1 << 17)
            for (size_t i = 0; i < 8; i++) {
                uint32_t t = (s_buf_y[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 0]) |
                             (s_buf_y[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 1] << 8) |
                             (s_buf_y[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 2] << 16);
                t >>= (threadIdx.x & 3) * 2;
                t &= 0x3FFFF;
                g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t) t;
            }
            // 20 bit per coefficient
#elif GAMMA1 == (1 << 19)
            for (size_t i = 0; i < 8; i++) {
                uint32_t t = (s_buf_y[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 0]) |
                             (s_buf_y[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 1] << 8) |
                             (s_buf_y[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 2] << 16);
                t >>= (threadIdx.x & 1) * 4;
                t &= 0xFFFFF;
                g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t) t;
            }
#endif
        }
    }
}

__global__ void compute_w_32t_kernel(int32_t *g_z, int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed,
                                     const int32_t *g_y, const int32_t *g_mat,
                                     const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
    __shared__ int32_t s_tmp[DILITHIUM_N + 32];

    int32_t regs[8];

    size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                     ? sign_idx_offset
                                     : blockIdx.x * temp_mem_pool_pitch;

    for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
        // make a copy of y, stored in z, perform ntt(z), for later use to calculate w = Ay
        const int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        for (size_t i = 0; i < 8; i++)
            regs[i] = g_y_l[i * 32 + threadIdx.x];
        ntt_inner(regs, s_tmp);
        for (size_t i = 0; i < 8; i++)
            g_z_l[threadIdx.x * 8 + i] = regs[i];
    }

    for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
        const int32_t *g_mat_k = g_mat + sign_idx_offset / sizeof(int32_t) + k * DILITHIUM_L * DILITHIUM_N;
        uint8_t *g_w1_packed_k = g_w1_packed + temp_idx_offset + k * POLYW1_PACKEDBYTES;
        int32_t *g_w0_k = g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        int32_t *g_w1_k = g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

        for (unsigned int i = 0; i < 8; ++i)
            regs[i] = 0;
        for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
            const int32_t *g_mat_k_l = g_mat_k + l * DILITHIUM_N;
            int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
            for (unsigned int i = 0; i < 8; ++i)
                regs[i] += gpu_montgomery_multiply(g_mat_k_l[i * 32 + threadIdx.x], g_z_l[i * 32 + threadIdx.x]);
        }

        for (unsigned int i = 0; i < 8; ++i)
            regs[i] = reduce32(regs[i]);

        for (unsigned int i = 0; i < 8; ++i)
            s_tmp[i * 32 + threadIdx.x] = regs[i];

        for (unsigned int i = 0; i < 8; ++i)
            regs[i] = s_tmp[threadIdx.x * 8 + i];

        invntt_inner(regs, s_tmp);

        for (unsigned int i = 0; i < 8; ++i) {
            regs[i] = caddq(regs[i]);
            regs[i] = decompose(&g_w0_k[32 * i + threadIdx.x], regs[i]);
        }

        for (unsigned int i = 0; i < 8; ++i) {
            s_tmp[32 * i + threadIdx.x] = regs[i];
            g_w1_k[32 * i + threadIdx.x] = regs[i];
        }
        __syncwarp();

#if GAMMA2 == (DILITHIUM_Q - 1) / 88
        for (unsigned int i = 0; i < 2; ++i) {
            g_w1_packed_k[6 * threadIdx.x + 3 * i + 0] = (s_tmp[8 * threadIdx.x + 4 * i + 0]) |
                                                         (s_tmp[8 * threadIdx.x + 4 * i + 1] << 6);
            g_w1_packed_k[6 * threadIdx.x + 3 * i + 1] = (s_tmp[8 * threadIdx.x + 4 * i + 1] >> 2) |
                                                         (s_tmp[8 * threadIdx.x + 4 * i + 2] << 4);
            g_w1_packed_k[6 * threadIdx.x + 3 * i + 2] = (s_tmp[8 * threadIdx.x + 4 * i + 2] >> 4) |
                                                         (s_tmp[8 * threadIdx.x + 4 * i + 3] << 2);
        }
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
        for (unsigned int i = 0; i < 4; ++i)
            g_w1_packed_k[4 * threadIdx.x + i] = (s_tmp[8 * threadIdx.x + 2 * i + 0]) |
                                                 (s_tmp[8 * threadIdx.x + 2 * i + 1] << 4);
#endif
    }
}

__global__ void compute_w_128t_kernel(int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed,
                                      const int32_t *g_y, const int32_t *g_mat,
                                      const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
    __shared__ int32_t s_tmp[DILITHIUM_N + 128];
    __shared__ int32_t s_zetas[DILITHIUM_N];

    s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
    s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
    __syncthreads();

    size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                     ? sign_idx_offset
                                     : blockIdx.x * temp_mem_pool_pitch;

    int32_t acc[DILITHIUM_K][2];

    for (auto &acc_k: acc) {
        acc_k[0] = 0;
        acc_k[1] = 0;
    }

    for (size_t l = 0; l < DILITHIUM_L; ++l) {
        const int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        int32_t reg0 = g_y_l[threadIdx.x];
        int32_t reg1 = g_y_l[128 + threadIdx.x];

        ntt_radix2_inner(reg0, reg1, s_tmp, s_zetas);

        for (size_t k = 0; k < DILITHIUM_K; ++k) {
            const int32_t *g_mat_k_l = g_mat + sign_idx_offset / sizeof(int32_t) + k * DILITHIUM_L * DILITHIUM_N + l * DILITHIUM_N;
            acc[k][0] += gpu_montgomery_multiply(g_mat_k_l[threadIdx.x], reg0);
            acc[k][1] += gpu_montgomery_multiply(g_mat_k_l[128 + threadIdx.x], reg1);
        }
        __syncthreads();
    }

    for (size_t k = 0; k < DILITHIUM_K; ++k) {

        acc[k][0] = reduce32(acc[k][0]);
        acc[k][1] = reduce32(acc[k][1]);

        intt_radix2_inner(acc[k][0], acc[k][1], s_tmp, s_zetas);

        acc[k][0] = caddq(acc[k][0]);
        acc[k][1] = caddq(acc[k][1]);

        // decompose w0, w1, then write to w0
        int32_t *g_w0_k = g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        acc[k][0] = decompose(&g_w0_k[threadIdx.x], acc[k][0]);
        acc[k][1] = decompose(&g_w0_k[128 + threadIdx.x], acc[k][1]);

        // write to w1
        int32_t *g_w1_k = g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        g_w1_k[threadIdx.x] = acc[k][0];
        g_w1_k[128 + threadIdx.x] = acc[k][1];

        // pack w1 into sig
        __syncthreads();
        s_tmp[threadIdx.x + (threadIdx.x >> 5)] = acc[k][0];
        s_tmp[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] = acc[k][1];
        __syncthreads();
        uint8_t *g_w1_packed_k = g_w1_packed + temp_idx_offset + k * POLYW1_PACKEDBYTES;
#if GAMMA2 == (DILITHIUM_Q - 1) / 88
        if (threadIdx.x < 64) {
            size_t offset = 4 * threadIdx.x + (threadIdx.x >> 3);
            g_w1_packed_k[3 * threadIdx.x + 0] = (s_tmp[offset + 0]) |
                                                 (s_tmp[offset + 1] << 6);
            g_w1_packed_k[3 * threadIdx.x + 1] = (s_tmp[offset + 1] >> 2) |
                                                 (s_tmp[offset + 2] << 4);
            g_w1_packed_k[3 * threadIdx.x + 2] = (s_tmp[offset + 2] >> 4) |
                                                 (s_tmp[offset + 3] << 2);
        }
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
        size_t offset = 2 * threadIdx.x + (threadIdx.x >> 4);
        g_w1_packed_k[threadIdx.x] = (s_tmp[offset + 0]) |
                                     (s_tmp[offset + 1] << 4);
#endif
        __syncthreads();
    }
}

__global__ void compute_cp_kernel(int32_t *g_cp, uint8_t *g_mu,
                                  uint8_t *g_seed,
                                  const uint8_t *g_ori_mu,
                                  const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
    __shared__ uint8_t s_buf2[SHAKE256_RATE];
    __shared__ int32_t s_tmp[DILITHIUM_N + 32];

    size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                     ? sign_idx_offset
                                     : blockIdx.x * temp_mem_pool_pitch;

    g_mu[temp_idx_offset + threadIdx.x] = g_ori_mu[sign_idx_offset + threadIdx.x];
    g_mu[temp_idx_offset + 32 + threadIdx.x] = g_ori_mu[sign_idx_offset + 32 + threadIdx.x];
    __syncwarp();
    shake<SHAKE256_RATE, 0x1f>(s_buf2, 1,
                               g_mu + temp_idx_offset,
                               CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
    __syncwarp();

    // write to seed in sig
    g_seed[temp_idx_offset + threadIdx.x] = s_buf2[threadIdx.x];

    // poly_challenge(&cp, sig);
    shake<SHAKE256_RATE, 0x1f>(s_buf2, 1, s_buf2, SEEDBYTES);
    __syncwarp();
    for (size_t i = 0; i < 8; i++) {
        s_tmp[i * 32 + threadIdx.x] = 0;
    }
    __syncwarp();
    if (threadIdx.x == 0) {
        uint64_t signs = reinterpret_cast<uint64_t *>(s_buf2)[0];
        size_t b;
        size_t pos = 8;
        for (size_t i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
            do {
                b = s_buf2[pos++];
            } while (b > i);

            s_tmp[i] = s_tmp[b];
            s_tmp[b] = 1 - 2 * (signs & 1);
            signs >>= 1;
        }
    }
    __syncwarp();
    for (size_t i = 0; i < 8; i++) {
        g_cp[temp_idx_offset / sizeof(int32_t) + i * 32 + threadIdx.x] = s_tmp[i * 32 + threadIdx.x];
    }
}

__global__ void compute_cp_opt_kernel(int32_t *g_cp, uint8_t *g_mu,
                                      uint8_t *g_seed,
                                      const uint8_t *g_ori_mu,
                                      const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch, size_t n_inputs) {
    __shared__ uint8_t s_buf2[4][SHAKE256_RATE];
    __shared__ int32_t s_tmp[4][DILITHIUM_N + 32];

    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (input_id < n_inputs) {

        uint8_t *s_buf2_y = s_buf2[threadIdx.y];
        int32_t *s_tmp_y = s_tmp[threadIdx.y];

        size_t sign_idx_offset = (g_exec_lut[input_id] >> 16) * sign_mem_pool_pitch;
        size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                         ? sign_idx_offset
                                         : input_id * temp_mem_pool_pitch;

        g_mu[temp_idx_offset + threadIdx.x] = g_ori_mu[sign_idx_offset + threadIdx.x];
        g_mu[temp_idx_offset + 32 + threadIdx.x] = g_ori_mu[sign_idx_offset + 32 + threadIdx.x];
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf2_y, 1,
                                   g_mu + temp_idx_offset,
                                   CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
        __syncwarp();

        // write to seed in sig
        g_seed[temp_idx_offset + threadIdx.x] = s_buf2_y[threadIdx.x];

        // poly_challenge(&cp, sig);
        shake<SHAKE256_RATE, 0x1f>(s_buf2_y, 1, s_buf2_y, SEEDBYTES);
        __syncwarp();
        for (size_t i = 0; i < 8; i++) {
            s_tmp_y[i * 32 + threadIdx.x] = 0;
        }
        __syncwarp();
        if (threadIdx.x == 0) {
            uint64_t signs = reinterpret_cast<uint64_t *>(s_buf2_y)[0];
            size_t pos = 8;
            size_t b;
            for (size_t i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
                do {
                    b = s_buf2_y[pos++];
                } while (b > i);

                s_tmp_y[i] = s_tmp_y[b];
                s_tmp_y[b] = 1 - 2 * (signs & 1);
                signs >>= 1;
            }
        }
        __syncwarp();
        for (size_t i = 0; i < 8; i++) {
            g_cp[temp_idx_offset / sizeof(int32_t) + i * 32 + threadIdx.x] = s_tmp_y[i * 32 + threadIdx.x];
        }
    }
}

__global__ void rej_loop_32t_kernel(
        const int32_t *g_y, int32_t *g_z, const int32_t *g_w0, const int32_t *g_w1, const int32_t *g_cp,
        uint8_t *g_z_packed, uint8_t *g_hint,
        uint8_t *g_done_lut,
        const int32_t *g_s1, const int32_t *g_s2, const int32_t *g_t0,
        const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
    __shared__ uint8_t s_hint[POLYVECH_PACKEDBYTES];

    __shared__ int32_t s_tmp[DILITHIUM_N + 32];
    __shared__ int32_t s_tmp2[DILITHIUM_N + 32];

    __shared__ int32_t s_cp[DILITHIUM_N + 32];

    int32_t regs[8];

    size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                     ? sign_idx_offset
                                     : blockIdx.x * temp_mem_pool_pitch;

    // poly_ntt(&cp);
    for (size_t i = 0; i < 8; i++)
        regs[i] = g_cp[temp_idx_offset / sizeof(int32_t) + i * 32 + threadIdx.x];
    ntt_inner(regs, s_tmp);
    for (size_t i = 0; i < 8; i++)
        s_cp[threadIdx.x * 8 + i] = regs[i];
    __syncwarp();

    // validation

    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += 32) {
        s_hint[i] = 0;
    }
    __syncwarp();

    for (size_t l = 0; l < DILITHIUM_L; ++l) {
        const int32_t *g_s1_l = g_s1 + sign_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        const int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;

        for (unsigned int i = 0; i < 8; ++i)
            // pointwise_montgomery
            s_tmp[32 * i + threadIdx.x] = gpu_montgomery_multiply(s_cp[32 * i + threadIdx.x], g_s1_l[32 * i + threadIdx.x]);

        for (size_t i = 0; i < 8; i++)
            regs[i] = s_tmp[threadIdx.x * 8 + i];
        // invntt_tomont
        invntt_inner(regs, s_tmp);

        for (size_t i = 0; i < 8; i++) {
            // add
            regs[i] += g_y_l[i * 32 + threadIdx.x];
            // reduce
            regs[i] = reduce32(regs[i]);

            // chknorm
            unsigned int rej_mask = __ballot_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(regs[i]));
            if (rej_mask) return;
            // write to tmp z
            g_z_l[i * 32 + threadIdx.x] = regs[i];
        }
    }

    unsigned int pos = 0;

    for (size_t k = 0; k < DILITHIUM_K; ++k) {
        const int32_t *g_s2_k = g_s2 + sign_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        const int32_t *g_t0_k = g_t0 + sign_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        const int32_t *g_w0_k = g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        const int32_t *g_w1_k = g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

        // tmp = cp * s2
        for (size_t i = 0; i < 8; ++i)
            // poly_pointwise_montgomery(&tmp, &cp, &s2.vec[i]);
            s_tmp[32 * i + threadIdx.x] = gpu_montgomery_multiply(s_cp[32 * i + threadIdx.x], g_s2_k[32 * i + threadIdx.x]);

        // regs = invntt(tmp)
        for (size_t i = 0; i < 8; i++)
            regs[i] = s_tmp[threadIdx.x * 8 + i];
        // poly_invntt_tomont(&tmp);
        invntt_inner(regs, s_tmp);

        for (size_t i = 0; i < 8; i++) {
            // poly_sub(&tmp, &w0.vec[i], &tmp);
            regs[i] = g_w0_k[32 * i + threadIdx.x] - regs[i];
            // poly_reduce(&tmp);
            regs[i] = reduce32(regs[i]);
            // poly_chknorm(&tmp, GAMMA2 - BETA)
            unsigned int rej_mask = __ballot_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(regs[i]));
            if (rej_mask) return;

            // write to tmp
            s_tmp[i * 32 + threadIdx.x] = regs[i];
        }

        // tmp2 = cp * t0; regs = invntt(tmp2)
        for (size_t i = 0; i < 8; i++)
            // poly_pointwise_montgomery(&tmp2, &cp, &t0.vec[i]);
            s_tmp2[32 * i + threadIdx.x] = gpu_montgomery_multiply(s_cp[32 * i + threadIdx.x], g_t0_k[32 * i + threadIdx.x]);

        for (size_t i = 0; i < 8; i++)
            // load to regs for invntt
            regs[i] = s_tmp2[threadIdx.x * 8 + i];

        // poly_invntt_tomont(&tmp2);
        invntt_inner(regs, s_tmp2);

        for (size_t i = 0; i < 8; i++) {
            // poly_reduce(&tmp2);
            regs[i] = reduce32(regs[i]);
            // poly_chknorm(&tmp2, GAMMA2)
            unsigned int rej_mask = __ballot_sync(0xFFFFFFFF, chknorm<GAMMA2>(regs[i]));
            if (rej_mask) return;

            // poly_add(&tmp, &tmp, &tmp2);
            regs[i] = regs[i] + s_tmp[32 * i + threadIdx.x];
        }

        // make hint
        for (size_t i = 0; i < 8; ++i) {
            // unsigned int n = poly_make_hint(&h.vec[i], &tmp, &w1.vec[i]);
            unsigned int hint_bit = make_hint(regs[i], g_w1_k[32 * i + threadIdx.x]);
            unsigned int hint_mask = __ballot_sync(0xFFFFFFFF, hint_bit);
            unsigned int inclusive_hint_mask = hint_mask << (31 - threadIdx.x);
            unsigned int n = __popc(hint_mask);
            unsigned int inclusive_n = __popc(inclusive_hint_mask);
            unsigned int rej_mask = __ballot_sync(0xFFFFFFFF, pos + inclusive_n > OMEGA);
            if (rej_mask) return;

            if (hint_bit) s_hint[pos + inclusive_n - 1] = 32 * i + threadIdx.x;
            __syncwarp();
            // for next loop
            pos += n;
        }
        // set last hint byte
        if (threadIdx.x == 0) s_hint[OMEGA + k] = pos;
        __syncwarp();
    }

    // write to packed z in signature
    for (size_t l = 0; l < DILITHIUM_L; ++l) {
        uint8_t *g_z_packed_l = g_z_packed + temp_idx_offset + l * POLYZ_PACKEDBYTES;
        int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
#if GAMMA1 == (1 << 17)
        for (size_t i = 0; i < 2; ++i) {
            size_t offset = 32 * i + threadIdx.x;

            uint32_t t0 = GAMMA1 - g_z_l[4 * offset + 0];
            uint32_t t1 = GAMMA1 - g_z_l[4 * offset + 1];
            uint32_t t2 = GAMMA1 - g_z_l[4 * offset + 2];
            uint32_t t3 = GAMMA1 - g_z_l[4 * offset + 3];

            g_z_packed_l[9 * offset + 0] = t0;
            g_z_packed_l[9 * offset + 1] = t0 >> 8;
            g_z_packed_l[9 * offset + 2] = t0 >> 16;
            g_z_packed_l[9 * offset + 2] |= t1 << 2;
            g_z_packed_l[9 * offset + 3] = t1 >> 6;
            g_z_packed_l[9 * offset + 4] = t1 >> 14;
            g_z_packed_l[9 * offset + 4] |= t2 << 4;
            g_z_packed_l[9 * offset + 5] = t2 >> 4;
            g_z_packed_l[9 * offset + 6] = t2 >> 12;
            g_z_packed_l[9 * offset + 6] |= t3 << 6;
            g_z_packed_l[9 * offset + 7] = t3 >> 2;
            g_z_packed_l[9 * offset + 8] = t3 >> 10;
        }
#elif GAMMA1 == (1 << 19)
        for (size_t i = 0; i < 4; ++i) {
            size_t offset = 32 * i + threadIdx.x;

            uint32_t t0 = GAMMA1 - g_z_l[2 * offset + 0];
            uint32_t t1 = GAMMA1 - g_z_l[2 * offset + 1];

            g_z_packed_l[5 * offset + 0] = t0;
            g_z_packed_l[5 * offset + 1] = t0 >> 8;
            g_z_packed_l[5 * offset + 2] = t0 >> 16;
            g_z_packed_l[5 * offset + 2] |= t1 << 4;
            g_z_packed_l[5 * offset + 3] = t1 >> 4;
            g_z_packed_l[5 * offset + 4] = t1 >> 12;
        }
#endif
    }

    // write to hint in signature
    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += 32) {
        g_hint[temp_idx_offset + i] = s_hint[i];
    }

    // set done flag
    if (threadIdx.x == 0) g_done_lut[blockIdx.x] = 1;
}

__global__ void rej_loop_128t_kernel(
        const int32_t *g_y, int32_t *g_z, const int32_t *g_w0, const int32_t *g_w1, const int32_t *g_cp,
        uint8_t *g_z_packed, uint8_t *g_hint,
        uint8_t *g_done_lut,
        const int32_t *g_s1, const int32_t *g_s2, const int32_t *g_t0,
        const uint32_t *g_exec_lut, size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
    __shared__ int32_t s_zetas[DILITHIUM_N];
    __shared__ int32_t s_ntt[DILITHIUM_N + 128];
    __shared__ uint8_t s_hint[POLYVECH_PACKEDBYTES];

    __shared__ int s_rej_flag;
    __shared__ size_t s_hint_count[4];
    __shared__ size_t s_hint_count_sum;

    // init rej flag with 0
    if (threadIdx.x == 0) s_rej_flag = 0;
    // cache zetas table to SMEM
    s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
    s_zetas[threadIdx.x + 128] = c_zetas[threadIdx.x + 128];
    // init hint with all 0
    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x) {
        s_hint[i] = 0;
    }
    __syncthreads();

    size_t lane_id = threadIdx.x & 0x1f;
    size_t warp_id = threadIdx.x >> 5;

    size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                     ? sign_idx_offset
                                     : blockIdx.x * temp_mem_pool_pitch;

    // poly_ntt(&cp);
    int32_t cp_regs0 = g_cp[temp_idx_offset / sizeof(int32_t) + threadIdx.x];
    int32_t cp_regs1 = g_cp[temp_idx_offset / sizeof(int32_t) + 128 + threadIdx.x];
    ntt_radix2_inner(cp_regs0, cp_regs1, s_ntt, s_zetas);

    // validation

    for (size_t l = 0; l < DILITHIUM_L; ++l) {
        const int32_t *g_s1_l = g_s1 + sign_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        const int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;

        int32_t cps1_reg0 = gpu_montgomery_multiply(cp_regs0, g_s1_l[threadIdx.x]);
        int32_t cps1_reg1 = gpu_montgomery_multiply(cp_regs1, g_s1_l[128 + threadIdx.x]);
        intt_radix2_inner(cps1_reg0, cps1_reg1, s_ntt, s_zetas);
        cps1_reg0 += g_y_l[threadIdx.x];
        cps1_reg1 += g_y_l[128 + threadIdx.x];
        cps1_reg0 = reduce32(cps1_reg0);
        cps1_reg1 = reduce32(cps1_reg1);
        // chknorm
        int rej_flag = __any_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(cps1_reg0));
        rej_flag += __any_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(cps1_reg1));
        if (lane_id == 0 && rej_flag) s_rej_flag = 1;
        __syncthreads();
        if (s_rej_flag) return;
        // write to tmp z
        g_z_l[threadIdx.x] = cps1_reg0;
        g_z_l[128 + threadIdx.x] = cps1_reg1;
    }

    unsigned int pos = 0;
    for (size_t k = 0; k < DILITHIUM_K; ++k) {
        const int32_t *g_s2_k = g_s2 + sign_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        const int32_t *g_t0_k = g_t0 + sign_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        const int32_t *g_w0_k = g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        const int32_t *g_w1_k = g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

        // tmp = cp * s2
        int32_t cps2_reg0 = gpu_montgomery_multiply(cp_regs0, g_s2_k[threadIdx.x]);
        int32_t cps2_reg1 = gpu_montgomery_multiply(cp_regs1, g_s2_k[128 + threadIdx.x]);
        intt_radix2_inner(cps2_reg0, cps2_reg1, s_ntt, s_zetas);

        // poly_sub(&tmp, &w0.vec[i], &tmp);
        cps2_reg0 = g_w0_k[threadIdx.x] - cps2_reg0;
        cps2_reg1 = g_w0_k[128 + threadIdx.x] - cps2_reg1;
        // poly_reduce(&tmp);
        cps2_reg0 = reduce32(cps2_reg0);
        cps2_reg1 = reduce32(cps2_reg1);
        // poly_chknorm(&tmp, GAMMA2 - BETA)
        int rej_flag = __any_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(cps2_reg0));
        rej_flag += __any_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(cps2_reg1));
        if (lane_id == 0 && rej_flag) s_rej_flag = 1;
        __syncthreads();
        if (s_rej_flag) return;

        // tmp2 = cp * t0; regs = invntt(tmp2)
        int32_t cpt0_reg0 = gpu_montgomery_multiply(cp_regs0, g_t0_k[threadIdx.x]);
        int32_t cpt0_reg1 = gpu_montgomery_multiply(cp_regs1, g_t0_k[128 + threadIdx.x]);
        intt_radix2_inner(cpt0_reg0, cpt0_reg1, s_ntt, s_zetas);

        // poly_reduce(&tmp2);
        cpt0_reg0 = reduce32(cpt0_reg0);
        cpt0_reg1 = reduce32(cpt0_reg1);

        // poly_chknorm(&tmp2, GAMMA2)
        rej_flag = __any_sync(0xFFFFFFFF, chknorm<GAMMA2>(cpt0_reg0));
        rej_flag += __any_sync(0xFFFFFFFF, chknorm<GAMMA2>(cpt0_reg1));
        if (lane_id == 0 && rej_flag) s_rej_flag = 1;
        __syncthreads();
        if (s_rej_flag) return;

        // poly_add(&tmp, &tmp, &tmp2);
        cpt0_reg0 += cps2_reg0;
        cpt0_reg1 += cps2_reg1;

        // make hint
        unsigned int hint_bit = make_hint(cpt0_reg0, g_w1_k[threadIdx.x]);
        unsigned int hint_mask = __ballot_sync(0xFFFFFFFF, hint_bit);
        unsigned int total_count_within_warp = __popc(hint_mask);
        if (lane_id == 0)
            s_hint_count[warp_id] = total_count_within_warp;
        __syncthreads();
        if (threadIdx.x == 0) {
            size_t count = 0;
            for (size_t i = 0; i < 4; ++i)
                count += s_hint_count[i];
            s_hint_count_sum = count;
        }
        __syncthreads();
        size_t total_count_within_block = s_hint_count_sum;
        if (pos + total_count_within_block > OMEGA) return;
        unsigned int inclusive_count_within_block = __popc(hint_mask << (31 - lane_id));
        for (size_t i = 0; i < warp_id; ++i)
            inclusive_count_within_block += s_hint_count[i];
        if (hint_bit)
            s_hint[pos + inclusive_count_within_block - 1] = threadIdx.x;
        __syncthreads();
        pos += total_count_within_block;

        hint_bit = make_hint(cpt0_reg1, g_w1_k[128 + threadIdx.x]);
        hint_mask = __ballot_sync(0xFFFFFFFF, hint_bit);
        total_count_within_warp = __popc(hint_mask);
        if (lane_id == 0)
            s_hint_count[warp_id] = total_count_within_warp;
        __syncthreads();
        if (threadIdx.x == 0) {
            size_t count = 0;
            for (size_t i = 0; i < 4; ++i)
                count += s_hint_count[i];
            s_hint_count_sum = count;
        }
        __syncthreads();
        total_count_within_block = s_hint_count_sum;
        if (pos + total_count_within_block > OMEGA) return;
        inclusive_count_within_block = __popc(hint_mask << (31 - lane_id));
        for (size_t i = 0; i < warp_id; ++i)
            inclusive_count_within_block += s_hint_count[i];
        if (hint_bit)
            s_hint[pos + inclusive_count_within_block - 1] = 128 + threadIdx.x;
        __syncthreads();
        pos += total_count_within_block;

        // set last hint byte
        if (threadIdx.x == 0) s_hint[OMEGA + k] = pos;
        __syncthreads();
    }

    // write to hint in signature
    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x) {
        g_hint[temp_idx_offset + i] = s_hint[i];
    }

    // write to packed z in signature
    for (size_t l = 0; l < DILITHIUM_L; ++l) {
        uint8_t *g_z_packed_l = g_z_packed + temp_idx_offset + l * POLYZ_PACKEDBYTES;
        const int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
#if GAMMA1 == (1 << 17)
        if (threadIdx.x < 64) {
            size_t i = threadIdx.x;
            uint32_t t[4];

            t[0] = GAMMA1 - g_z_l[4 * i + 0];
            t[1] = GAMMA1 - g_z_l[4 * i + 1];
            t[2] = GAMMA1 - g_z_l[4 * i + 2];
            t[3] = GAMMA1 - g_z_l[4 * i + 3];

            g_z_packed_l[9 * i + 0] = t[0];
            g_z_packed_l[9 * i + 1] = t[0] >> 8;
            g_z_packed_l[9 * i + 2] = t[0] >> 16;
            g_z_packed_l[9 * i + 2] |= t[1] << 2;
            g_z_packed_l[9 * i + 3] = t[1] >> 6;
            g_z_packed_l[9 * i + 4] = t[1] >> 14;
            g_z_packed_l[9 * i + 4] |= t[2] << 4;
            g_z_packed_l[9 * i + 5] = t[2] >> 4;
            g_z_packed_l[9 * i + 6] = t[2] >> 12;
            g_z_packed_l[9 * i + 6] |= t[3] << 6;
            g_z_packed_l[9 * i + 7] = t[3] >> 2;
            g_z_packed_l[9 * i + 8] = t[3] >> 10;
        }
#elif GAMMA1 == (1 << 19)
        size_t i = threadIdx.x;
        uint32_t t[2];

        t[0] = GAMMA1 - g_z_l[2 * i + 0];
        t[1] = GAMMA1 - g_z_l[2 * i + 1];

        g_z_packed_l[5 * i + 0] = t[0];
        g_z_packed_l[5 * i + 1] = t[0] >> 8;
        g_z_packed_l[5 * i + 2] = t[0] >> 16;
        g_z_packed_l[5 * i + 2] |= t[1] << 4;
        g_z_packed_l[5 * i + 3] = t[1] >> 4;
        g_z_packed_l[5 * i + 4] = t[1] >> 12;
#endif
    }

    // set done flag
    if (threadIdx.x == 0) g_done_lut[blockIdx.x] = 1;
}

__global__ void sig_copy_kernel(uint8_t *d_sig, size_t d_sign_mem_pool_pitch,
                                const uint8_t *d_temp_sig, size_t d_temp_mem_pool_pitch,
                                const copy_lut_element *d_copy_lut) {
    copy_lut_element e = d_copy_lut[blockIdx.x];
    size_t exec_idx_offset = e.exec_idx * d_temp_mem_pool_pitch;
    size_t sign_idx_offset = e.sign_idx * d_sign_mem_pool_pitch;
    for (size_t i = threadIdx.x; i < CRYPTO_BYTES; i += blockDim.x) {
        d_sig[sign_idx_offset + i] = d_temp_sig[exec_idx_offset + i];
    }
}

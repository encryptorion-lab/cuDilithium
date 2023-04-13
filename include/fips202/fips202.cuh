/*
 * Copyright (c) 2021 Tatsuki Ono
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/mit-license.php
 *
 * Modified by [Anonymous] in 2023.
 */

#pragma once

#include "params.h"
#include <cstdint>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136

constexpr unsigned keccak_rounds = 24;

extern __device__ const uint64_t keccak_f_round_constants[keccak_rounds];
extern __device__ const int8_t offset_constants[32];

class gpu_keccak {
private:
    char4 params0_{};
    char4 params1_{};

    __device__ uint64_t f1600_state_permute(uint64_t state) const;

public:
    __device__ gpu_keccak();

    template<unsigned int rate, uint8_t p>
    __device__ uint64_t absorb_once(const uint8_t *in, size_t in_len);

    template<unsigned int rate>
    __device__ void squeezeblocks(uint64_t &state, uint8_t *out, size_t n_blocks);

    template<unsigned int rate>
    __device__ void shake(uint8_t *out, size_t out_len,
                          const uint8_t *in, size_t in_len,
                          uint8_t *tmp_shared) {

        const size_t n_blocks = out_len / rate;
        const size_t blocks_len = n_blocks * rate;

        uint64_t state = absorb_once<rate, 0x1f>(in, in_len);
        squeezeblocks<rate>(state, out, n_blocks);

        out_len -= blocks_len;

        if (out_len > 0) {
            out += blocks_len;
            squeezeblocks<rate>(state, tmp_shared, 1);
            __syncwarp();

            for (unsigned i = threadIdx.x; i < out_len; i += 32) {
                out[i] = tmp_shared[i];
            }
        }
    }

    template<unsigned int rate>
    __device__ __inline__ void notmp_shake(std::uint8_t *out_blocksized, std::size_t out_blocks,
                                           const std::uint8_t *in, std::size_t in_len) {
        uint64_t state = absorb_once<rate, 0x1f>(in, in_len);
        squeezeblocks<rate>(state, out_blocksized, out_blocks);
    }
};

template<unsigned int rate>
__global__ void shake_kernel(uint8_t *out, size_t out_pitch, size_t out_len,
                             const uint8_t *in, size_t in_pitch, size_t in_len,
                             unsigned n_inputs) {
    extern __shared__ uint8_t tmp_shared[];

    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (input_id < n_inputs) {
        uint8_t *tmp_shared_ptr = tmp_shared + rate * threadIdx.y;
        out += out_pitch * input_id;
        in += in_pitch * input_id;
        gpu_keccak keccak;
        keccak.shake<rate>(out, out_len, in, in_len, tmp_shared_ptr);
    }
}

template<unsigned int rate>
__global__ void notmp_shake_kernel(uint8_t *out, size_t out_pitch, size_t out_len,
                                   const uint8_t *in, size_t in_pitch, size_t in_len,
                                   unsigned n_inputs) {
    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (input_id < n_inputs) {
        out += out_pitch * input_id;
        in += in_pitch * input_id;

        size_t out_blocks = (out_len + (rate - 1)) / rate;
        gpu_keccak keccak;
        keccak.notmp_shake<rate>(out, out_blocks, in, in_len);
    }
}

__device__ __forceinline__ uint64_t state_permute(uint64_t state, char4 params) {
    auto rol = [](uint64_t a, unsigned b, unsigned c) noexcept -> uint64_t {
        return (a << b) ^ (a >> c);
    };

    const int8_t x = threadIdx.x % 5;
    const int8_t theta1 = params.x;
    const int8_t theta2 = params.y;
    const int8_t theta4 = (threadIdx.x + 4) % 5;
    const int8_t offset = params.z;
    const int8_t rp = params.w;
    const int8_t chi1 = threadIdx.x - x + theta1;// 5y + (x + 1) % 5
    const int8_t chi2 = threadIdx.x - x + theta2;// 5y + (x + 2) % 5

    uint64_t a = state;

    if (threadIdx.x < 25) {
        const unsigned active_mask = __activemask();
        uint64_t c;

        for (unsigned round = 0; round < keccak_rounds; ++round) {
            // theta
            c = __shfl_sync(active_mask, a, x + 0) ^
                __shfl_sync(active_mask, a, x + 5) ^
                __shfl_sync(active_mask, a, x + 10) ^
                __shfl_sync(active_mask, a, x + 15) ^
                __shfl_sync(active_mask, a, x + 20);
            a = a ^ (__shfl_sync(active_mask, c, theta4) ^
                     rol(__shfl_sync(active_mask, c, theta1), 1, 63));

            // rho and pi
            c = __shfl_sync(active_mask, rol(a, offset, 64 - offset), rp);

            // chi
            a = c ^ ((~__shfl_sync(active_mask, c, chi1)) &
                     __shfl_sync(active_mask, c, chi2));

            // iota
            if (threadIdx.x == 0) a = a ^ __ldg(&keccak_f_round_constants[round]);
        }
    }

    return a;
}

#include <cstdio>

template<unsigned int rate, uint8_t p>
__global__ void notmp_shake_new_kernel(uint8_t *out, size_t out_pitch, size_t out_len,
                                       const uint8_t *in, size_t in_pitch, size_t in_len,
                                       unsigned n_inputs) {
    const int8_t x = threadIdx.x % 5;
    const int8_t y = threadIdx.x / 5;
    const int8_t theta1 = (x + 1) % 5;// (x + 1) mod 5
    const int8_t theta2 = (x + 2) % 5;// (x + 2) mod 5
    const int8_t offset = __ldg(&offset_constants[threadIdx.x]);
    const int8_t rp = (x + 3 * y) % 5 + 5 * x;

    char4 params = make_char4(theta1, theta2, offset, rp);

    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (input_id < n_inputs) {
        out += out_pitch * input_id;
        in += in_pitch * input_id;

        size_t out_blocks = (out_len + (rate - 1)) / rate;

        in += 8 * threadIdx.x;

        uint64_t state = 0;

        while (in_len >= rate) {
            if (threadIdx.x < rate / 8) {
                auto in64 = reinterpret_cast<const uint64_t *>(in);
                state ^= *in64;
            }
            state = state_permute(state, params);
            in_len -= rate;
            in += rate;
        }

        if (threadIdx.x <= in_len / 8) {
            auto in64 = reinterpret_cast<const uint64_t *>(in);
            uint64_t in64_value = *in64;
            if (threadIdx.x == in_len / 8) {
                in64_value &= ((1ULL << (8 * (in_len % 8))) - 1);
            }
            state ^= in64_value;
        }

        if (in_len / 8 == threadIdx.x) {
            state ^= static_cast<uint64_t>(p) << ((in_len % 8) * 8);
        }

        if ((rate - 1) / 8 == threadIdx.x) {
            state ^= 0x8000'0000'0000'0000ULL;
        }

        // squeeze blocks
        while (out_blocks > 0) {
            state = state_permute(state, params);
            if (threadIdx.x < rate / 8) {
                auto out64 = reinterpret_cast<uint64_t *>(out);
                out64[threadIdx.x] = state;
            }
            out += rate;
            --out_blocks;
        }
    }
}

template<unsigned int rate, uint8_t p>
__device__ __forceinline__ void shake(uint8_t *out, size_t out_blocks,
                                      const uint8_t *in, size_t in_len) {
    const int8_t x = threadIdx.x % 5;
    const int8_t y = threadIdx.x / 5;
    const int8_t theta1 = (x + 1) % 5;// (x + 1) mod 5
    const int8_t theta2 = (x + 2) % 5;// (x + 2) mod 5
    const int8_t offset = __ldg(&offset_constants[threadIdx.x]);
    const int8_t rp = (x + 3 * y) % 5 + 5 * x;

    char4 params = make_char4(theta1, theta2, offset, rp);

    //    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    in += 8 * threadIdx.x;

    uint64_t state = 0;

    while (in_len >= rate) {
        if (threadIdx.x < rate / 8) {
            auto in64 = reinterpret_cast<const uint64_t *>(in);
            state ^= *in64;
        }
        state = state_permute(state, params);
        in_len -= rate;
        in += rate;
    }

    if (threadIdx.x <= in_len / 8) {
        auto in64 = reinterpret_cast<const uint64_t *>(in);
        uint64_t in64_value = *in64;
        if (threadIdx.x == in_len / 8) {
            in64_value &= ((1ULL << (8 * (in_len % 8))) - 1);
        }
        state ^= in64_value;
    }

    if (in_len / 8 == threadIdx.x) {
        state ^= static_cast<uint64_t>(p) << ((in_len % 8) * 8);
    }

    if ((rate - 1) / 8 == threadIdx.x) {
        state ^= 0x8000'0000'0000'0000ULL;
    }

    __syncwarp();

    // squeeze blocks
    while (out_blocks > 0) {
        state = state_permute(state, params);
        if (threadIdx.x < rate / 8) {
            auto out64 = reinterpret_cast<uint64_t *>(out);
            out64[threadIdx.x] = state;
        }
        out += rate;
        --out_blocks;
    }
}

#include "keccak.cuh"
#include <cstdio>


__device__ const uint64_t keccak_f_round_constants[keccak_rounds] = {
        0x0000'0000'0000'0001ULL, 0x0000'0000'0000'8082ULL,
        0x8000'0000'0000'808aULL, 0x8000'0000'8000'8000ULL,
        0x0000'0000'0000'808bULL, 0x0000'0000'8000'0001ULL,
        0x8000'0000'8000'8081ULL, 0x8000'0000'0000'8009ULL,
        0x0000'0000'0000'008aULL, 0x0000'0000'0000'0088ULL,
        0x0000'0000'8000'8009ULL, 0x0000'0000'8000'000aULL,
        0x0000'0000'8000'808bULL, 0x8000'0000'0000'008bULL,
        0x8000'0000'0000'8089ULL, 0x8000'0000'0000'8003ULL,
        0x8000'0000'0000'8002ULL, 0x8000'0000'0000'0080ULL,
        0x0000'0000'0000'800aULL, 0x8000'0000'8000'000aULL,
        0x8000'0000'8000'8081ULL, 0x8000'0000'0000'8080ULL,
        0x0000'0000'8000'0001ULL, 0x8000'0000'8000'8008ULL};

__device__ const int8_t offset_constants[32] = {
        /* x = 0, 1, 2, 3, 4 */
        0, 1, 62, 28, 27, // y = 0
        36, 44, 6, 55, 20,// y = 1
        3, 10, 43, 25, 39,// y = 2
        41, 45, 15, 21, 8,// y = 3
        18, 2, 61, 56, 14 // y = 4
};

__device__ gpu_keccak::gpu_keccak() {
    int8_t x = threadIdx.x % 5;
    int8_t y = threadIdx.x / 5;
    int8_t theta1 = (x + 1) % 5;// (x + 1) mod 5
    int8_t theta4 = (x + 4) % 5;// (x - 1) mod 5
    int8_t offset = offset_constants[threadIdx.x];
    int8_t rp = (x + 3 * y) % 5 + 5 * x;
    int8_t chi1 = threadIdx.x - x + theta1;     // 5y + (x + 1) % 5
    int8_t chi2 = threadIdx.x - x + (x + 2) % 5;// 5y + (x + 2) % 5

    params0_ = make_char4(x, y, theta1, theta4);
    params1_ = make_char4(offset, rp, chi1, chi2);
}

__device__ uint64_t gpu_keccak::f1600_state_permute(uint64_t state) const {
    auto rol = [](uint64_t a, unsigned b, unsigned c) noexcept -> uint64_t {
        return (a << b) ^ (a >> c);
    };

    const int8_t x = params0_.x;
    // const int8_t y = params0_.y;
    const int8_t theta1 = params0_.z;
    const int8_t theta4 = params0_.w;
    const int8_t offset = params1_.x;
    const int8_t rp = params1_.y;
    const int8_t chi1 = params1_.z;
    const int8_t chi2 = params1_.w;

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

/*************************************************
* Name:        keccak::absorb_once
*
* Description: Absorb step of Keccak;
*              non-incremental, starts by zeroeing the state.
*
* Arguments:   - unsigned int rate: rate in bytes (e.g., 168 for SHAKE128)
*              - const uint8_t *in: pointer to input to be absorbed into s
*              - size_t in_len: length of input in bytes
*              - uint8_t p: domain-separation byte for different Keccak-derived functions
**************************************************/
template __device__ uint64_t gpu_keccak::absorb_once<SHAKE128_RATE, 0x1f>(const uint8_t *in, size_t in_len);
template __device__ uint64_t gpu_keccak::absorb_once<SHAKE256_RATE, 0x1f>(const uint8_t *in, size_t in_len);
template<unsigned int rate, uint8_t p>
[[nodiscard]] __device__ uint64_t gpu_keccak::absorb_once(const uint8_t *in, size_t in_len) {
    auto load64 = [](const uint8_t x[8]) -> uint64_t {
        uint64_t r = static_cast<uint64_t>(x[0]) << 0;
        r |= static_cast<uint64_t>(x[1]) << 8;
        r |= static_cast<uint64_t>(x[2]) << 16;
        r |= static_cast<uint64_t>(x[3]) << 24;
        r |= static_cast<uint64_t>(x[4]) << 32;
        r |= static_cast<uint64_t>(x[5]) << 40;
        r |= static_cast<uint64_t>(x[6]) << 48;
        r |= static_cast<uint64_t>(x[7]) << 56;
        return r;
    };

    in += 8 * threadIdx.x;

    uint64_t state = 0;

    while (in_len >= rate) {
        if (threadIdx.x < rate / 8) {
            state ^= load64(in);
        }
        state = f1600_state_permute(state);
        in_len -= rate;
        in += rate;
    }

    for (unsigned i = 0; i < 8; ++i) {
        state ^= (8 * threadIdx.x + i < in_len)
                         ? static_cast<uint64_t>(in[i]) << (8 * i)
                         : 0;
    }

    if (in_len / 8 == threadIdx.x) {
        state ^= static_cast<uint64_t>(p) << ((in_len % 8) * 8);
    }

    if ((rate - 1) / 8 == threadIdx.x) {
        state ^= 0x8000'0000'0000'0000ULL;
    }

    return state;
}

/*************************************************
* Name:        keccak::squeezeblocks
*
* Description: Squeeze step of Keccak. Squeezes full blocks of r bytes each.
*              Modifies the state. Can be called multiple times to keep
*              squeezing, i.e., is incremental. Assumes zero bytes of current
*              block have already been squeezed.
*
* Arguments:   - uint8_t *out: pointer to output blocks
*              - size_t n_blocks: number of blocks to be squeezed (written to out)
*              - unsigned int rate: rate in bytes (e.g., 168 for SHAKE128)
**************************************************/
template __device__ void gpu_keccak::squeezeblocks<SHAKE128_RATE>(uint64_t &state, uint8_t *out, size_t n_blocks);
template __device__ void gpu_keccak::squeezeblocks<SHAKE256_RATE>(uint64_t &state, uint8_t *out, size_t n_blocks);
template<unsigned int rate>
__device__ void gpu_keccak::squeezeblocks(uint64_t &state, uint8_t *out, size_t n_blocks) {
    auto store64 = [](uint8_t x[8], uint64_t s) {
        x[0] = s >> 0;
        x[1] = s >> 8;
        x[2] = s >> 16;
        x[3] = s >> 24;
        x[4] = s >> 32;
        x[5] = s >> 40;
        x[6] = s >> 48;
        x[7] = s >> 56;
    };

    while (n_blocks > 0) {
        state = f1600_state_permute(state);
        if (threadIdx.x < rate / 8) {
            store64(&out[8 * threadIdx.x], state);
        }
        out += rate;
        --n_blocks;
    }
}

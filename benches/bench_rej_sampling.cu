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
#include "fips202/fips202.cuh"

#define NTESTS 10000

#define POLY_UNIFORM_NBLOCKS ((768 + SHAKE128_RATE - 1) / SHAKE128_RATE)

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

__global__ void rej_sampling(int32_t *g_mat, size_t mem_pool_pitch) {
    __shared__ uint8_t s_buf[4][POLY_UNIFORM_NBLOCKS * SHAKE128_RATE];
    uint8_t *s_buf_y = s_buf[threadIdx.y];
    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    shake<SHAKE128_RATE, 0x1f>(s_buf_y, POLY_UNIFORM_NBLOCKS, s_buf_y, SEEDBYTES + 2);
    __syncwarp();

    size_t ctr = 0;
    auto g_poly = g_mat + input_id * mem_pool_pitch / sizeof(int32_t);
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

typedef struct {
    uint64_t s[25];
    unsigned int pos;
} keccak_state;

#define NROUNDS 24
#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset)))

static uint64_t load64(const uint8_t x[8]) {
    unsigned int i;
    uint64_t r = 0;

    for (i = 0; i < 8; i++)
        r |= (uint64_t) x[i] << 8 * i;

    return r;
}

static void store64(uint8_t x[8], uint64_t u) {
    unsigned int i;

    for (i = 0; i < 8; i++)
        x[i] = u >> 8 * i;
}

/* Keccak round constants */
const uint64_t KeccakF_RoundConstants[NROUNDS] = {
        (uint64_t) 0x0000000000000001ULL,
        (uint64_t) 0x0000000000008082ULL,
        (uint64_t) 0x800000000000808aULL,
        (uint64_t) 0x8000000080008000ULL,
        (uint64_t) 0x000000000000808bULL,
        (uint64_t) 0x0000000080000001ULL,
        (uint64_t) 0x8000000080008081ULL,
        (uint64_t) 0x8000000000008009ULL,
        (uint64_t) 0x000000000000008aULL,
        (uint64_t) 0x0000000000000088ULL,
        (uint64_t) 0x0000000080008009ULL,
        (uint64_t) 0x000000008000000aULL,
        (uint64_t) 0x000000008000808bULL,
        (uint64_t) 0x800000000000008bULL,
        (uint64_t) 0x8000000000008089ULL,
        (uint64_t) 0x8000000000008003ULL,
        (uint64_t) 0x8000000000008002ULL,
        (uint64_t) 0x8000000000000080ULL,
        (uint64_t) 0x000000000000800aULL,
        (uint64_t) 0x800000008000000aULL,
        (uint64_t) 0x8000000080008081ULL,
        (uint64_t) 0x8000000000008080ULL,
        (uint64_t) 0x0000000080000001ULL,
        (uint64_t) 0x8000000080008008ULL
};

static void KeccakF1600_StatePermute(uint64_t state[25]) {
    int round;

    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;
    uint64_t BCa, BCe, BCi, BCo, BCu;
    uint64_t Da, De, Di, Do, Du;
    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;

    //copyFromState(A, state)
    Aba = state[0];
    Abe = state[1];
    Abi = state[2];
    Abo = state[3];
    Abu = state[4];
    Aga = state[5];
    Age = state[6];
    Agi = state[7];
    Ago = state[8];
    Agu = state[9];
    Aka = state[10];
    Ake = state[11];
    Aki = state[12];
    Ako = state[13];
    Aku = state[14];
    Ama = state[15];
    Ame = state[16];
    Ami = state[17];
    Amo = state[18];
    Amu = state[19];
    Asa = state[20];
    Ase = state[21];
    Asi = state[22];
    Aso = state[23];
    Asu = state[24];

    for (round = 0; round < NROUNDS; round += 2) {
        //    prepareTheta
        BCa = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        BCe = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        BCi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        BCo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        BCu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        //thetaRhoPiChiIotaPrepareTheta(round, A, E)
        Da = BCu ^ ROL(BCe, 1);
        De = BCa ^ ROL(BCi, 1);
        Di = BCe ^ ROL(BCo, 1);
        Do = BCi ^ ROL(BCu, 1);
        Du = BCo ^ ROL(BCa, 1);

        Aba ^= Da;
        BCa = Aba;
        Age ^= De;
        BCe = ROL(Age, 44);
        Aki ^= Di;
        BCi = ROL(Aki, 43);
        Amo ^= Do;
        BCo = ROL(Amo, 21);
        Asu ^= Du;
        BCu = ROL(Asu, 14);
        Eba = BCa ^ ((~BCe) & BCi);
        Eba ^= (uint64_t) KeccakF_RoundConstants[round];
        Ebe = BCe ^ ((~BCi) & BCo);
        Ebi = BCi ^ ((~BCo) & BCu);
        Ebo = BCo ^ ((~BCu) & BCa);
        Ebu = BCu ^ ((~BCa) & BCe);

        Abo ^= Do;
        BCa = ROL(Abo, 28);
        Agu ^= Du;
        BCe = ROL(Agu, 20);
        Aka ^= Da;
        BCi = ROL(Aka, 3);
        Ame ^= De;
        BCo = ROL(Ame, 45);
        Asi ^= Di;
        BCu = ROL(Asi, 61);
        Ega = BCa ^ ((~BCe) & BCi);
        Ege = BCe ^ ((~BCi) & BCo);
        Egi = BCi ^ ((~BCo) & BCu);
        Ego = BCo ^ ((~BCu) & BCa);
        Egu = BCu ^ ((~BCa) & BCe);

        Abe ^= De;
        BCa = ROL(Abe, 1);
        Agi ^= Di;
        BCe = ROL(Agi, 6);
        Ako ^= Do;
        BCi = ROL(Ako, 25);
        Amu ^= Du;
        BCo = ROL(Amu, 8);
        Asa ^= Da;
        BCu = ROL(Asa, 18);
        Eka = BCa ^ ((~BCe) & BCi);
        Eke = BCe ^ ((~BCi) & BCo);
        Eki = BCi ^ ((~BCo) & BCu);
        Eko = BCo ^ ((~BCu) & BCa);
        Eku = BCu ^ ((~BCa) & BCe);

        Abu ^= Du;
        BCa = ROL(Abu, 27);
        Aga ^= Da;
        BCe = ROL(Aga, 36);
        Ake ^= De;
        BCi = ROL(Ake, 10);
        Ami ^= Di;
        BCo = ROL(Ami, 15);
        Aso ^= Do;
        BCu = ROL(Aso, 56);
        Ema = BCa ^ ((~BCe) & BCi);
        Eme = BCe ^ ((~BCi) & BCo);
        Emi = BCi ^ ((~BCo) & BCu);
        Emo = BCo ^ ((~BCu) & BCa);
        Emu = BCu ^ ((~BCa) & BCe);

        Abi ^= Di;
        BCa = ROL(Abi, 62);
        Ago ^= Do;
        BCe = ROL(Ago, 55);
        Aku ^= Du;
        BCi = ROL(Aku, 39);
        Ama ^= Da;
        BCo = ROL(Ama, 41);
        Ase ^= De;
        BCu = ROL(Ase, 2);
        Esa = BCa ^ ((~BCe) & BCi);
        Ese = BCe ^ ((~BCi) & BCo);
        Esi = BCi ^ ((~BCo) & BCu);
        Eso = BCo ^ ((~BCu) & BCa);
        Esu = BCu ^ ((~BCa) & BCe);

        //    prepareTheta
        BCa = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        BCe = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        BCi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        BCo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        BCu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
        Da = BCu ^ ROL(BCe, 1);
        De = BCa ^ ROL(BCi, 1);
        Di = BCe ^ ROL(BCo, 1);
        Do = BCi ^ ROL(BCu, 1);
        Du = BCo ^ ROL(BCa, 1);

        Eba ^= Da;
        BCa = Eba;
        Ege ^= De;
        BCe = ROL(Ege, 44);
        Eki ^= Di;
        BCi = ROL(Eki, 43);
        Emo ^= Do;
        BCo = ROL(Emo, 21);
        Esu ^= Du;
        BCu = ROL(Esu, 14);
        Aba = BCa ^ ((~BCe) & BCi);
        Aba ^= (uint64_t) KeccakF_RoundConstants[round + 1];
        Abe = BCe ^ ((~BCi) & BCo);
        Abi = BCi ^ ((~BCo) & BCu);
        Abo = BCo ^ ((~BCu) & BCa);
        Abu = BCu ^ ((~BCa) & BCe);

        Ebo ^= Do;
        BCa = ROL(Ebo, 28);
        Egu ^= Du;
        BCe = ROL(Egu, 20);
        Eka ^= Da;
        BCi = ROL(Eka, 3);
        Eme ^= De;
        BCo = ROL(Eme, 45);
        Esi ^= Di;
        BCu = ROL(Esi, 61);
        Aga = BCa ^ ((~BCe) & BCi);
        Age = BCe ^ ((~BCi) & BCo);
        Agi = BCi ^ ((~BCo) & BCu);
        Ago = BCo ^ ((~BCu) & BCa);
        Agu = BCu ^ ((~BCa) & BCe);

        Ebe ^= De;
        BCa = ROL(Ebe, 1);
        Egi ^= Di;
        BCe = ROL(Egi, 6);
        Eko ^= Do;
        BCi = ROL(Eko, 25);
        Emu ^= Du;
        BCo = ROL(Emu, 8);
        Esa ^= Da;
        BCu = ROL(Esa, 18);
        Aka = BCa ^ ((~BCe) & BCi);
        Ake = BCe ^ ((~BCi) & BCo);
        Aki = BCi ^ ((~BCo) & BCu);
        Ako = BCo ^ ((~BCu) & BCa);
        Aku = BCu ^ ((~BCa) & BCe);

        Ebu ^= Du;
        BCa = ROL(Ebu, 27);
        Ega ^= Da;
        BCe = ROL(Ega, 36);
        Eke ^= De;
        BCi = ROL(Eke, 10);
        Emi ^= Di;
        BCo = ROL(Emi, 15);
        Eso ^= Do;
        BCu = ROL(Eso, 56);
        Ama = BCa ^ ((~BCe) & BCi);
        Ame = BCe ^ ((~BCi) & BCo);
        Ami = BCi ^ ((~BCo) & BCu);
        Amo = BCo ^ ((~BCu) & BCa);
        Amu = BCu ^ ((~BCa) & BCe);

        Ebi ^= Di;
        BCa = ROL(Ebi, 62);
        Ego ^= Do;
        BCe = ROL(Ego, 55);
        Eku ^= Du;
        BCi = ROL(Eku, 39);
        Ema ^= Da;
        BCo = ROL(Ema, 41);
        Ese ^= De;
        BCu = ROL(Ese, 2);
        Asa = BCa ^ ((~BCe) & BCi);
        Ase = BCe ^ ((~BCi) & BCo);
        Asi = BCi ^ ((~BCo) & BCu);
        Aso = BCo ^ ((~BCu) & BCa);
        Asu = BCu ^ ((~BCa) & BCe);
    }

    //copyToState(state, A)
    state[0] = Aba;
    state[1] = Abe;
    state[2] = Abi;
    state[3] = Abo;
    state[4] = Abu;
    state[5] = Aga;
    state[6] = Age;
    state[7] = Agi;
    state[8] = Ago;
    state[9] = Agu;
    state[10] = Aka;
    state[11] = Ake;
    state[12] = Aki;
    state[13] = Ako;
    state[14] = Aku;
    state[15] = Ama;
    state[16] = Ame;
    state[17] = Ami;
    state[18] = Amo;
    state[19] = Amu;
    state[20] = Asa;
    state[21] = Ase;
    state[22] = Asi;
    state[23] = Aso;
    state[24] = Asu;
}

static void keccak_init(uint64_t s[25]) {
    unsigned int i;
    for (i = 0; i < 25; i++)
        s[i] = 0;
}

static unsigned int keccak_absorb(uint64_t s[25],
                                  unsigned int pos,
                                  unsigned int r,
                                  const uint8_t *in,
                                  size_t inlen) {
    unsigned int i;

    while (pos + inlen >= r) {
        for (i = pos; i < r; i++)
            s[i / 8] ^= (uint64_t) *in++ << 8 * (i % 8);
        inlen -= r - pos;
        KeccakF1600_StatePermute(s);
        pos = 0;
    }

    for (i = pos; i < pos + inlen; i++)
        s[i / 8] ^= (uint64_t) *in++ << 8 * (i % 8);

    return i;
}

static void keccak_finalize(uint64_t s[25], unsigned int pos, unsigned int r, uint8_t p) {
    s[pos / 8] ^= (uint64_t) p << 8 * (pos % 8);
    s[r / 8 - 1] ^= 1ULL << 63;
}

static unsigned int keccak_squeeze(uint8_t *out,
                                   size_t outlen,
                                   uint64_t s[25],
                                   unsigned int pos,
                                   unsigned int r) {
    unsigned int i;

    while (outlen) {
        if (pos == r) {
            KeccakF1600_StatePermute(s);
            pos = 0;
        }
        for (i = pos; i < r && i < pos + outlen; i++)
            *out++ = s[i / 8] >> 8 * (i % 8);
        outlen -= i - pos;
        pos = i;
    }

    return pos;
}

static void keccak_absorb_once(uint64_t s[25],
                               unsigned int r,
                               const uint8_t *in,
                               size_t inlen,
                               uint8_t p) {
    unsigned int i;

    for (i = 0; i < 25; i++)
        s[i] = 0;

    while (inlen >= r) {
        for (i = 0; i < r / 8; i++)
            s[i] ^= load64(in + 8 * i);
        in += r;
        inlen -= r;
        KeccakF1600_StatePermute(s);
    }

    for (i = 0; i < inlen; i++)
        s[i / 8] ^= (uint64_t) in[i] << 8 * (i % 8);

    s[i / 8] ^= (uint64_t) p << 8 * (i % 8);
    s[(r - 1) / 8] ^= 1ULL << 63;
}

static void keccak_squeezeblocks(uint8_t *out,
                                 size_t nblocks,
                                 uint64_t s[25],
                                 unsigned int r) {
    unsigned int i;

    while (nblocks) {
        KeccakF1600_StatePermute(s);
        for (i = 0; i < r / 8; i++)
            store64(out + 8 * i, s[i]);
        out += r;
        nblocks -= 1;
    }
}

void shake128_init(keccak_state *state) {
    keccak_init(state->s);
    state->pos = 0;
}

void shake128_absorb(keccak_state *state, const uint8_t *in, size_t inlen) {
    state->pos = keccak_absorb(state->s, state->pos, SHAKE128_RATE, in, inlen);
}

void shake128_finalize(keccak_state *state) {
    keccak_finalize(state->s, state->pos, SHAKE128_RATE, 0x1F);
    state->pos = SHAKE128_RATE;
}

void shake128_squeeze(uint8_t *out, size_t outlen, keccak_state *state) {
    state->pos = keccak_squeeze(out, outlen, state->s, state->pos, SHAKE128_RATE);
}

void shake128_absorb_once(keccak_state *state, const uint8_t *in, size_t inlen) {
    keccak_absorb_once(state->s, SHAKE128_RATE, in, inlen, 0x1F);
    state->pos = SHAKE128_RATE;
}

void shake128_squeezeblocks(uint8_t *out, size_t nblocks, keccak_state *state) {
    keccak_squeezeblocks(out, nblocks, state->s, SHAKE128_RATE);
}

static unsigned int rej_uniform(std::vector<size_t> &vec_accept_index,
                                unsigned int len,
                                const uint8_t *buf,
                                unsigned int buflen) {
    unsigned int ctr, pos;
    uint32_t t;

    ctr = pos = 0;
    while (ctr < len && pos + 3 <= buflen) {
        t = buf[pos++];
        t |= (uint32_t) buf[pos++] << 8;
        t |= (uint32_t) buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < DILITHIUM_Q) {
            ctr++;
            vec_accept_index.push_back(pos - 3);
        }
    }

    return ctr;
}

void dilithium_shake128_stream_init(keccak_state *state, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
    uint8_t t[2];
    t[0] = nonce;
    t[1] = nonce >> 8;

    shake128_init(state);
    shake128_absorb(state, seed, SEEDBYTES);
    shake128_absorb(state, t, 2);
    shake128_finalize(state);
}

typedef struct {
    int32_t coeffs[DILITHIUM_N];
} poly;

void poly_uniform(std::vector<uint8_t> &buf,
                  std::vector<size_t> &vec_accept_index,
                  const uint8_t seed[SEEDBYTES],
                  uint16_t nonce) {
    unsigned int i, ctr, off;
    unsigned int buflen = POLY_UNIFORM_NBLOCKS * SHAKE128_RATE;
    keccak_state state;

    dilithium_shake128_stream_init(&state, seed, nonce);
    shake128_squeezeblocks(buf.data(), POLY_UNIFORM_NBLOCKS, &state);

    ctr = rej_uniform(vec_accept_index, DILITHIUM_N, buf.data(), buflen);

    while (ctr < DILITHIUM_N) {
        off = buflen % 3;
        for (i = 0; i < off; ++i)
            buf[i] = buf[buflen - off + i];

        shake128_squeezeblocks(buf.data() + off, 1, &state);
        buflen = SHAKE128_RATE + off;
        ctr += rej_uniform(vec_accept_index, DILITHIUM_N - ctr, buf.data(), buflen);
    }
}

void test_our_rej() {
    int32_t *d_mat;
    size_t pitch;
    cudaMallocPitch(&d_mat, &pitch, sizeof(int32_t) * DILITHIUM_N, NTESTS);
    CUDATimer timer_rej("our rej sampling");
    timer_rej.start();
    rej_sampling<<<(NTESTS + 3) / 4, dim3(32, 4)>>>(d_mat, pitch);
    timer_rej.stop();
}

__global__ void seo_rej_sampling(int32_t *g_mat, size_t g_mat_pitch, const uint8_t *g_buf, size_t g_buf_pitch,
                                 const size_t *g_accept_index, size_t g_accept_index_pitch) {
    const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

    size_t ctr = 0;
    auto g_poly = g_mat + input_id * g_mat_pitch / sizeof(int32_t);
    auto g_per_buf = g_buf + input_id * g_buf_pitch / sizeof(uint8_t);
    auto g_per_accept_index = g_accept_index + input_id * g_accept_index_pitch / sizeof(size_t);
    for (size_t round = 0; round < 8; round++) {
        size_t pos = g_per_accept_index[ctr + threadIdx.x];
        uint8_t t0 = g_per_buf[pos + 0];
        uint8_t t1 = g_per_buf[pos + 1];
        uint8_t t2 = g_per_buf[pos + 2] & 0x7F;
        uint32_t t = t0 | (t1 << 8) | (t2 << 16);
        g_poly[ctr + threadIdx.x] = t;
        ctr += 32;
    }
}

void test_seo_rej() {
    int32_t *d_mat;
    size_t d_mat_pitch;
    cudaMallocPitch(&d_mat, &d_mat_pitch, sizeof(int32_t) * DILITHIUM_N, NTESTS);

    uint8_t *d_buf;
    size_t d_buf_pitch;
    cudaMallocPitch(&d_buf, &d_buf_pitch, sizeof(uint8_t) * (POLY_UNIFORM_NBLOCKS * SHAKE128_RATE + 2), NTESTS);

    size_t *d_accept_index;
    size_t d_accept_index_pitch;
    cudaMallocPitch(&d_accept_index, &d_accept_index_pitch, sizeof(size_t) * DILITHIUM_N, NTESTS);

    CUDATimer timer_rej_1("seo rej offline");
    CUDATimer timer_rej_2("seo rej online");
    timer_rej_1.start();
    for (size_t i = 0; i < NTESTS; i++) {
        uint8_t seed[SEEDBYTES];
        uint16_t nonce = 0;
        std::vector<size_t> vec_accept_index;
        std::vector<uint8_t> buf(POLY_UNIFORM_NBLOCKS * SHAKE128_RATE + 2);
        poly_uniform(buf, vec_accept_index, seed, nonce);
        if (vec_accept_index.size() != DILITHIUM_N)
            throw std::logic_error("vec_accept_index should be N");
        cudaMemcpy(d_buf + i * d_buf_pitch / sizeof(uint8_t), buf.data(), sizeof(uint8_t) * POLY_UNIFORM_NBLOCKS * SHAKE128_RATE,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_accept_index + i * d_accept_index_pitch / sizeof(size_t), vec_accept_index.data(),
                   sizeof(size_t) * DILITHIUM_N, cudaMemcpyHostToDevice);
    }
    timer_rej_1.stop();

    timer_rej_2.start();
    seo_rej_sampling<<<(NTESTS + 3) / 4, dim3(32, 4)>>>(d_mat, d_mat_pitch, d_buf, d_buf_pitch, d_accept_index,
                                                        d_accept_index_pitch);
    timer_rej_2.stop();
}

int main() {
    test_our_rej();
    test_seo_rej();
    return 0;
}

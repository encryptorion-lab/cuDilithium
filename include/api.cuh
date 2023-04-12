#pragma once

#include <cstddef>
#include <cstdint>

#include "params.h"
#include "util.cuh"

#define ALIGN_TO_256_BYTES(x) ((((x) + 255) / 256) * 256)

#define crypto_sign_keypair DILITHIUM_NAMESPACE(keypair)
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

#define crypto_sign_signature DILITHIUM_NAMESPACE(signature)
int crypto_sign_signature(uint8_t *sig, size_t sig_pitch, size_t *siglen,
                          const uint8_t *m, size_t m_pitch, size_t mlen,
                          const uint8_t *sk,
                          uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch,
                          uint8_t *d_temp_mem_pool, size_t d_temp_mem_pool_pitch,
                          task_lut &lut, size_t exec_threshold, size_t batch_size = 1, cudaStream_t stream = nullptr);

#define crypto_sign_verify DILITHIUM_NAMESPACE(verify)
void crypto_sign_verify(int *ret,
                        const uint8_t *sig, size_t sig_pitch, size_t siglen,
                        const uint8_t *m, size_t m_pitch, size_t mlen,
                        const uint8_t *pk,
                        uint8_t *d_verify_mem_pool, size_t verify_mem_pool_pitch,
                        size_t batch_size = 1, cudaStream_t stream = nullptr);

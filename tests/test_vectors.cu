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

#include <cstdio>

#include "api.cuh"
#include "fips202/fips202.cuh"

extern "C" {
#include "randombytes.h"
#include "fips202/fips202.h"
}

#define MLEN 32
#define NVECTORS 10000

void randombytes(uint8_t *out, size_t outlen, uint64_t ctr) {
    unsigned int i;
    uint8_t buf[8];

    for (i = 0; i < 8; ++i)
        buf[i] = ctr >> 8 * i;

    shake128(out, outlen, buf, 8);
}

int main() {
    char filename[10];
    sprintf(filename, "tvecs%d", DILITHIUM_MODE);
    FILE *fp = fopen(filename, "w");

    unsigned int i, j;
    uint8_t buf[CRYPTO_SECRETKEYBYTES];
    size_t siglen;

    uint8_t *h_pk, *h_sk, *h_m, *h_sig;
    cudaMallocHost(&h_pk, NVECTORS * CRYPTO_PUBLICKEYBYTES);
    cudaMallocHost(&h_sk, NVECTORS * CRYPTO_SECRETKEYBYTES);
    cudaMallocHost(&h_m, NVECTORS * MLEN);
    cudaMallocHost(&h_sig, NVECTORS * CRYPTO_BYTES);

    int *h_ret;
    cudaMallocHost(&h_ret, NVECTORS * sizeof(int));

    for (i = 0; i < NVECTORS; ++i) {
        randombytes(h_m + i * MLEN, MLEN, i * 2);
    }

    uint8_t *d_keypair_mem_pool;
    size_t keypair_mem_pool_pitch;
    size_t byte_size_per_keypair = ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +// pk
                                   ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES); // sk

    size_t mem_size_per_keypair = byte_size_per_keypair +                     // align to 256 bytes
                                  DILITHIUM_K * DILITHIUM_N * sizeof(int32_t);// d_as

    cudaMallocPitch(&d_keypair_mem_pool, &keypair_mem_pool_pitch, mem_size_per_keypair, NVECTORS);

    crypto_sign_keypair(h_pk, h_sk, d_keypair_mem_pool, keypair_mem_pool_pitch, NVECTORS);
    cudaDeviceSynchronize();

    cudaFree(d_keypair_mem_pool);

    // gpu sign

    uint8_t *d_sign_mem_pool;
    size_t d_sign_mem_pool_pitch;
    size_t byte_size = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                                           // sig
                       ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES) +                                  // sk
                       ALIGN_TO_256_BYTES(SEEDBYTES) +                                              // d_rho
                       ALIGN_TO_256_BYTES(SEEDBYTES + MLEN) +                                       // d_tr || d_m
                       ALIGN_TO_256_BYTES(SEEDBYTES + CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +
                       // d_key || d_mu || d_w1_packed
                       ALIGN_TO_256_BYTES(CRHBYTES);                                                // d_rhoprime

    size_t sign_mem_size = byte_size +                                                // align to 256 bytes
                           DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +// mat
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // d_s1
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // d_y
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // d_z
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_t0
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_s2
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_w1
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_w0
                           DILITHIUM_N * sizeof(int32_t);                             // d_cp

    cudaMallocPitch(&d_sign_mem_pool, &d_sign_mem_pool_pitch, sign_mem_size, NVECTORS);

    // sign temp pool

    size_t exec_threshold = 2048;

    task_lut lut{};
    lut.sign_lut.resize(NVECTORS);
    cudaMallocHost(&lut.h_exec_lut, sizeof(uint32_t) * exec_threshold);
    cudaMallocHost(&lut.h_done_lut, sizeof(uint8_t) * exec_threshold);
    cudaMallocHost(&lut.h_copy_lut, sizeof(copy_lut_element) * exec_threshold);
    cudaMalloc(&lut.d_exec_lut, sizeof(uint32_t) * exec_threshold);
    cudaMalloc(&lut.d_done_lut, sizeof(uint8_t) * exec_threshold);
    cudaMalloc(&lut.d_copy_lut, sizeof(copy_lut_element) * exec_threshold);

    uint8_t *d_temp_mem_pool;
    size_t d_temp_mem_pool_pitch;
    size_t byte_size_per_sign_temp = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                              // sig
                                     ALIGN_TO_256_BYTES(
                                             CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);// d_mu || d_w1_packed

    size_t temp_mem_size = byte_size_per_sign_temp +                    // align to 256 bytes
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +// d_y
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +// d_z
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +// d_w0
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +// d_w1
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t); // d_cp

    cudaMallocPitch(&d_temp_mem_pool, &d_temp_mem_pool_pitch, temp_mem_size, exec_threshold);

    crypto_sign_signature(h_sig, CRYPTO_BYTES, &siglen, h_m, MLEN, MLEN, h_sk,
                          d_sign_mem_pool, d_sign_mem_pool_pitch,
                          d_temp_mem_pool, d_temp_mem_pool_pitch,
                          lut, exec_threshold, NVECTORS);
    cudaDeviceSynchronize();

    cudaFree(d_sign_mem_pool);
    cudaFree(d_temp_mem_pool);
    cudaFreeHost(lut.h_exec_lut);
    cudaFreeHost(lut.h_done_lut);
    cudaFreeHost(lut.h_copy_lut);
    cudaFree(lut.d_exec_lut);
    cudaFree(lut.d_done_lut);
    cudaFree(lut.d_copy_lut);

    // gpu verify

    uint8_t *d_verify_mem_pool;
    size_t d_verify_mem_pool_pitch;
    size_t byte_size_per_verify = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                               // sig
                                  ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +
                                  // pk (rho || t1(packed))
                                  ALIGN_TO_256_BYTES(SEEDBYTES + MLEN) +                           // muprime || m
                                  ALIGN_TO_256_BYTES(CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +
                                  // mu || w1_prime_packed
                                  ALIGN_TO_256_BYTES(SEEDBYTES);                                   // c_tilde

    size_t mem_size_per_verify = byte_size_per_verify +                                     // align to 4 bytes
                                 DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +// mat
                                 DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // z
                                 DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // t1
                                 DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // w1prime
                                 DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // h
                                 DILITHIUM_N * sizeof(int32_t) +                            // cp
                                 sizeof(int);                                               // ret

    cudaMallocPitch(&d_verify_mem_pool, &d_verify_mem_pool_pitch, mem_size_per_verify, NVECTORS);

    crypto_sign_verify(h_ret,
                       h_sig, CRYPTO_BYTES, CRYPTO_BYTES,
                       h_m, MLEN, MLEN,
                       h_pk,
                       d_verify_mem_pool, d_verify_mem_pool_pitch, NVECTORS);
    cudaDeviceSynchronize();

    cudaFree(d_verify_mem_pool);

    for (i = 0; i < NVECTORS; ++i) {
        if (h_ret[i]) {
            fprintf(fp, "Signature verification failed! %u\n", i);
            fprintf(stderr, "Signature verification failed! %u\n", i);
        }
    }

    for (i = 0; i < NVECTORS; ++i) {
        fprintf(fp, "count = %u\n", i);

        fprintf(fp, "m = ");
        for (j = 0; j < MLEN; ++j)
            fprintf(fp, "%02x", h_m[i * MLEN + j]);
        fprintf(fp, "\n");

        shake256(buf, 32, h_pk + i * CRYPTO_PUBLICKEYBYTES, CRYPTO_PUBLICKEYBYTES);
        fprintf(fp, "pk = ");
        for (j = 0; j < 32; ++j)
            fprintf(fp, "%02x", buf[j]);
        fprintf(fp, "\n");

        shake256(buf, 32, h_sk + i * CRYPTO_SECRETKEYBYTES, CRYPTO_SECRETKEYBYTES);
        fprintf(fp, "sk = ");
        for (j = 0; j < 32; ++j)
            fprintf(fp, "%02x", buf[j]);
        fprintf(fp, "\n");

        shake256(buf, 32, h_sig + i * CRYPTO_BYTES, CRYPTO_BYTES);
        fprintf(fp, "sig = ");
        for (j = 0; j < 32; ++j)
            fprintf(fp, "%02x", buf[j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
    cudaFreeHost(h_pk);
    cudaFreeHost(h_sk);
    cudaFreeHost(h_m);
    cudaFreeHost(h_sig);
    return 0;
}

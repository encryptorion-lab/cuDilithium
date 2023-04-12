#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <omp.h>
#include <vector>

extern "C" {
#include "randombytes.h"
}

#include "api.cuh"
#include "util.cuh"

#define MLEN 32

int bench_cudilithium(size_t batch_size, size_t exec_threshold, size_t n_streams) {
    size_t smlen;

    uint8_t *h_pk, *h_sk, *h_m, *h_sm;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_pk, batch_size * CRYPTO_PUBLICKEYBYTES));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_sk, batch_size * CRYPTO_SECRETKEYBYTES));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_m, batch_size * MLEN));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_sm, batch_size * CRYPTO_BYTES));

    int *h_ret;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_ret, batch_size * sizeof(int)));

    // Keypair

    uint8_t *d_keypair_mem_pool;
    size_t d_keypair_mem_pool_pitch;
    size_t byte_size_per_keypair = ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +// pk
                                   ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES); // sk

    size_t mem_size_per_keypair = byte_size_per_keypair +                     // align to 256 bytes
                                  DILITHIUM_K * DILITHIUM_N * sizeof(int32_t);// d_as

    CHECK_CUDA_ERROR(cudaMallocPitch(&d_keypair_mem_pool, &d_keypair_mem_pool_pitch, mem_size_per_keypair, batch_size));

    // Sign

    uint8_t *d_sign_mem_pool;
    size_t d_sign_mem_pool_pitch;
    size_t byte_size_per_sign = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                                           // sig
                                ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES) +                                  // sk
                                ALIGN_TO_256_BYTES(SEEDBYTES) +                                              // d_rho
                                ALIGN_TO_256_BYTES(SEEDBYTES + MLEN) +
                                // d_tr || d_m
                                ALIGN_TO_256_BYTES(SEEDBYTES + CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +
                                // d_key || d_mu || d_w1_packed
                                ALIGN_TO_256_BYTES(
                                        CRHBYTES);                                                // d_rhoprime

    size_t mem_size_per_sign = byte_size_per_sign +                                       // align to 8 bytes
                               DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +// mat
                               DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // d_s1
                               DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // d_y
                               DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // d_z
                               DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_t0
                               DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_s2
                               DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_w1
                               DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // d_w0
                               DILITHIUM_N * sizeof(int32_t);                             // d_cp

    CHECK_CUDA_ERROR(cudaMallocPitch(&d_sign_mem_pool, &d_sign_mem_pool_pitch, mem_size_per_sign, batch_size));

    // sign temp pool

    task_lut lut{};
    lut.sign_lut.resize(batch_size);
    CHECK_CUDA_ERROR(cudaMallocHost(&lut.h_exec_lut, sizeof(uint32_t) * exec_threshold));
    CHECK_CUDA_ERROR(cudaMallocHost(&lut.h_done_lut, sizeof(uint8_t) * exec_threshold));
    CHECK_CUDA_ERROR(cudaMallocHost(&lut.h_copy_lut, sizeof(copy_lut_element) * exec_threshold));
    CHECK_CUDA_ERROR(cudaMalloc(&lut.d_exec_lut, sizeof(uint32_t) * exec_threshold));
    CHECK_CUDA_ERROR(cudaMalloc(&lut.d_done_lut, sizeof(uint8_t) * exec_threshold));
    CHECK_CUDA_ERROR(cudaMalloc(&lut.d_copy_lut, sizeof(copy_lut_element) * exec_threshold));

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

    CHECK_CUDA_ERROR(cudaMallocPitch(&d_temp_mem_pool, &d_temp_mem_pool_pitch, temp_mem_size, exec_threshold));

    // Verify

    uint8_t *d_verify_mem_pool;
    size_t d_verify_mem_pool_pitch;
    size_t byte_size_per_verify = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                               // sig
                                  ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +
                                  // pk (rho || t1(packed))
                                  ALIGN_TO_256_BYTES(SEEDBYTES + MLEN) +                           // muprime || m
                                  ALIGN_TO_256_BYTES(CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +
                                  // mu || w1_prime_packed
                                  ALIGN_TO_256_BYTES(SEEDBYTES);                                   // c_tilde

    size_t mem_size_per_verify = byte_size_per_verify +                                     // align to 256 bytes
                                 DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +// mat
                                 DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +              // z
                                 DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // t1
                                 DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // w1prime
                                 DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +              // h
                                 DILITHIUM_N * sizeof(int32_t) +                            // cp
                                 sizeof(int);                                               // ret

    CHECK_CUDA_ERROR(cudaMallocPitch(&d_verify_mem_pool, &d_verify_mem_pool_pitch, mem_size_per_verify, batch_size));

    //    ChronoTimer timer_verify_batch("verify batch");
    //    ChronoTimer timer_sign_batch("sign_batch");
    //    ChronoTimer timer_keypair_batch("keypair batch");

    for (size_t i = 0; i < batch_size; ++i)
        randombytes(h_m + i * MLEN, MLEN);

    /*
    for (size_t test_idx = 0; test_idx < 1; test_idx++) {
        //        timer_keypair_batch.start();
        crypto_sign_keypair(h_pk, h_sk, d_keypair_mem_pool, d_keypair_mem_pool_pitch, batch_size);
        cudaDeviceSynchronize();
        //        timer_keypair_batch.stop();

        //        timer_sign_batch.start();
        crypto_sign_signature(
                h_sm, CRYPTO_BYTES, &smlen,
                h_m, MLEN, MLEN,
                h_sk,
                d_sign_mem_pool, d_sign_mem_pool_pitch,
                d_temp_mem_pool, d_temp_mem_pool_pitch,
                lut, exec_threshold, batch_size);
        cudaDeviceSynchronize();
        //        timer_sign_batch.stop();

        timer_verify_batch.start();
        crypto_sign_verify(
                h_ret,
                h_sm, CRYPTO_BYTES, CRYPTO_BYTES,
                h_m, MLEN, MLEN,
                h_pk,
                d_verify_mem_pool, d_verify_mem_pool_pitch,
                batch_size);
        cudaDeviceSynchronize();
        timer_verify_batch.stop();

        for (size_t i = 0; i < batch_size; ++i) {
            if (h_ret[i]) {
                fprintf(stderr, "Verification failed: %zu\n", i);
                return -1;
            }
        }
    }
    */
    ChronoTimer timer_verify_stream("verify stream");
    ChronoTimer timer_sign_stream("sign stream");
    ChronoTimer timer_keypair_stream("keypair stream");

    // cuda stream init
    std::vector<cudaStream_t> streams(n_streams);
    for (auto &stream: streams) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    for (size_t test_idx = 0; test_idx < 1000; test_idx++) {
        timer_keypair_stream.start();
        for (size_t i = 0; i < n_streams; ++i) {
            size_t keypair_stream_batch_size = batch_size / n_streams;
            size_t last_keypair_stream_batch_size = keypair_stream_batch_size;
            last_keypair_stream_batch_size += (i == n_streams - 1) ? (batch_size % n_streams) : 0;
            crypto_sign_keypair(h_pk + i * keypair_stream_batch_size * CRYPTO_PUBLICKEYBYTES,
                                h_sk + i * keypair_stream_batch_size * CRYPTO_SECRETKEYBYTES,
                                d_keypair_mem_pool + i * keypair_stream_batch_size * d_keypair_mem_pool_pitch,
                                d_keypair_mem_pool_pitch, last_keypair_stream_batch_size, streams[i]);
        }
        cudaDeviceSynchronize();
        timer_keypair_stream.stop();

        timer_sign_stream.start();
#pragma omp parallel for default(none) shared(h_sm, smlen, h_m, h_sk, d_sign_mem_pool, d_sign_mem_pool_pitch, d_temp_mem_pool, d_temp_mem_pool_pitch, lut, streams, batch_size, exec_threshold, n_streams)
        for (size_t i = 0; i < n_streams; ++i) {
            size_t sign_stream_batch_size = batch_size / n_streams;
            size_t last_sign_stream_batch_size = sign_stream_batch_size;
            last_sign_stream_batch_size += (i == n_streams - 1) ? (batch_size % n_streams) : 0;
            size_t stream_exec_threshold = exec_threshold / n_streams;
            size_t last_stream_exec_threshold = stream_exec_threshold;
            last_stream_exec_threshold += (i == n_streams - 1) ? (exec_threshold % n_streams) : 0;

            task_lut lut_i = {};
            lut_i.sign_lut.resize(last_sign_stream_batch_size);
            lut_i.h_exec_lut = lut.h_exec_lut + i * stream_exec_threshold;
            lut_i.h_done_lut = lut.h_done_lut + i * stream_exec_threshold;
            lut_i.h_copy_lut = lut.h_copy_lut + i * stream_exec_threshold;
            lut_i.d_exec_lut = lut.d_exec_lut + i * stream_exec_threshold;
            lut_i.d_done_lut = lut.d_done_lut + i * stream_exec_threshold;
            lut_i.d_copy_lut = lut.d_copy_lut + i * stream_exec_threshold;
            crypto_sign_signature(
                    h_sm + i * sign_stream_batch_size * CRYPTO_BYTES, CRYPTO_BYTES, &smlen,
                    h_m + i * sign_stream_batch_size * MLEN, MLEN, MLEN,
                    h_sk + i * sign_stream_batch_size * CRYPTO_SECRETKEYBYTES,
                    d_sign_mem_pool + i * sign_stream_batch_size * d_sign_mem_pool_pitch, d_sign_mem_pool_pitch,
                    d_temp_mem_pool + i * stream_exec_threshold * d_temp_mem_pool_pitch, d_temp_mem_pool_pitch,
                    lut_i, last_stream_exec_threshold, last_sign_stream_batch_size, streams[i]);
        }
        cudaDeviceSynchronize();
        timer_sign_stream.stop();

        timer_verify_stream.start();
        for (size_t i = 0; i < n_streams; ++i) {
            size_t verify_stream_batch_size = batch_size / n_streams;
            size_t last_verify_stream_batch_size = verify_stream_batch_size;
            last_verify_stream_batch_size += (i == n_streams - 1) ? (batch_size % n_streams) : 0;
            crypto_sign_verify(
                    h_ret + i * verify_stream_batch_size,
                    h_sm + i * verify_stream_batch_size * CRYPTO_BYTES, CRYPTO_BYTES, CRYPTO_BYTES,
                    h_m + i * verify_stream_batch_size * MLEN, MLEN, MLEN,
                    h_pk + i * verify_stream_batch_size * CRYPTO_PUBLICKEYBYTES,
                    d_verify_mem_pool + i * verify_stream_batch_size * d_verify_mem_pool_pitch, d_verify_mem_pool_pitch,
                    last_verify_stream_batch_size, streams[i]);
        }
        cudaDeviceSynchronize();
        timer_verify_stream.stop();

        for (size_t i = 0; i < batch_size; ++i) {
            if (h_ret[i]) {
                fprintf(stderr, "Verification failed: %zu\n", i);
                return -1;
            }
        }
    }
    // cuda stream destroy
    for (auto &stream: streams) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }

    cudaFree(d_keypair_mem_pool);

    cudaFree(d_sign_mem_pool);
    cudaFree(d_temp_mem_pool);
    cudaFreeHost(lut.h_exec_lut);
    cudaFreeHost(lut.h_done_lut);
    cudaFreeHost(lut.h_copy_lut);
    cudaFree(lut.d_exec_lut);
    cudaFree(lut.d_done_lut);
    cudaFree(lut.d_copy_lut);

    cudaFree(d_verify_mem_pool);

    cudaFreeHost(h_pk);
    cudaFreeHost(h_sk);
    cudaFreeHost(h_m);
    cudaFreeHost(h_sm);
    cudaFreeHost(h_ret);

    return 0;
}

int main() {
    std::vector<size_t> batch_sizes = {10000};
    std::vector<size_t> exec_thresholds = {2512};
    std::vector<size_t> v_n_streams = {10};
    //    for (size_t i = 1; i <= 16; i += 1)
    //        v_n_streams.push_back(i);
    //    auto header_fmt = boost::format("%10s%10s%20s%10s%20s%20s%20s%20s") % "threshold" % "batch" % "function" % "trials" % "min" % "mean" % "median" % "stddev.";
    auto header_fmt =
            boost::format("%10s%20s%10s%20s%20s%20s%20s") % "n_streams" % "function" % "trials" % "min" % "mean" %
            "median" % "stddev.";
    std::cout << header_fmt << std::endl;
    for (auto &batch_size: batch_sizes) {
        for (auto &exec_threshold: exec_thresholds) {
            for (auto &n_streams: v_n_streams) {
                //                auto fmt = boost::format("%10d%10d") % exec_threshold % batch_size;
                //                auto fmt = boost::format("%10d") % n_streams;
                //                std::cout << fmt;
                bench_cudilithium(batch_size, exec_threshold, n_streams);
            }
        }
    }
    return 0;
}

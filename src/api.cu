#include "api.cuh"

#include "keccak.cuh"
#include "keypair.cuh"
#include "params.h"
#include "sign.cuh"
#include "util.cuh"
#include "verify.cuh"

int crypto_sign_keypair(uint8_t *pk, uint8_t *sk,
                        uint8_t *d_keypair_mem_pool, size_t keypair_mem_pool_pitch,
                        size_t batch_size, cudaStream_t stream, size_t rand_index) {
    size_t byte_size = ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +// pk
                       ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES); // sk

    // pk
    uint8_t *d_pk = d_keypair_mem_pool;
    uint8_t *d_pk_rho = d_pk + 0;
    uint8_t *d_pk_t1 = d_pk_rho + SEEDBYTES;

    // sk
    uint8_t *d_sk = d_keypair_mem_pool + ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES);
    uint8_t *d_sk_rho = d_sk + 0;
    uint8_t *d_sk_key = d_sk_rho + SEEDBYTES;
    uint8_t *d_sk_tr = d_sk_key + SEEDBYTES;
    uint8_t *d_sk_s1_packed = d_sk_tr + SEEDBYTES;
    uint8_t *d_sk_s2_packed = d_sk_s1_packed + DILITHIUM_L * POLYETA_PACKEDBYTES;
    uint8_t *d_sk_t0_packed = d_sk_s2_packed + DILITHIUM_K * POLYETA_PACKEDBYTES;

    // As
    auto *d_t1 = (int32_t *) (d_keypair_mem_pool + byte_size);
    cudaMemset2DAsync(d_t1, keypair_mem_pool_pitch, 0, DILITHIUM_K * DILITHIUM_N * sizeof(int32_t), batch_size, stream);

    gpu_keypair<<<batch_size, 32, 0, stream>>>(d_pk_rho, d_pk_t1,
                                               d_sk_rho, d_sk_key, d_sk_tr,
                                               d_sk_s1_packed, d_sk_s2_packed, d_sk_t0_packed,
                                               d_t1, keypair_mem_pool_pitch, rand_index);

    cudaMemcpy2DAsync(pk, CRYPTO_PUBLICKEYBYTES, d_pk, keypair_mem_pool_pitch, CRYPTO_PUBLICKEYBYTES, batch_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpy2DAsync(sk, CRYPTO_SECRETKEYBYTES, d_sk, keypair_mem_pool_pitch, CRYPTO_SECRETKEYBYTES, batch_size, cudaMemcpyDeviceToHost, stream);

    return 0;
}

/**
 * @brief Computes signature.
 * @param [out] sig pointer to output signature (of length CRYPTO_BYTES)
 * @param sig_pitch output signature pitch
 * @param siglen pointer to output length of signature
 * @param [in] m pointer to message to be signed
 * @param m_pitch message pitch
 * @param mlen length of message
 * @param [in] sk pointer to bit-packed secret key
 * @param batch_size (optional) batch size, default is 1
 * @return 0 (success)
 */
int crypto_sign_signature(uint8_t *sig, size_t sig_pitch, size_t *siglen,
                          const uint8_t *m, size_t m_pitch, size_t mlen,
                          const uint8_t *sk,
                          uint8_t *d_sign_mem_pool, size_t d_sign_mem_pool_pitch,
                          uint8_t *d_temp_mem_pool, size_t d_temp_mem_pool_pitch,
                          task_lut &lut, size_t exec_threshold, size_t batch_size, cudaStream_t stream) {

    uint8_t *d_sig = d_sign_mem_pool + 0;
    uint8_t *d_seed = d_sig;
    uint8_t *d_z_packed = d_seed + SEEDBYTES;
    uint8_t *d_hint = d_z_packed + DILITHIUM_L * POLYZ_PACKEDBYTES;

    uint8_t *d_sk = d_sig + ALIGN_TO_256_BYTES(CRYPTO_BYTES);
    uint8_t *d_sk_rho = d_sk;
    uint8_t *d_sk_key = d_sk_rho + SEEDBYTES;
    uint8_t *d_sk_tr = d_sk_key + SEEDBYTES;
    uint8_t *d_sk_s1_packed = d_sk_tr + SEEDBYTES;
    uint8_t *d_sk_s2_packed = d_sk_s1_packed + DILITHIUM_L * POLYETA_PACKEDBYTES;
    uint8_t *d_sk_t0_packed = d_sk_s2_packed + DILITHIUM_K * POLYETA_PACKEDBYTES;

    uint8_t *d_rho = d_sk + ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES);

    // tr || m
    uint8_t *d_tr = d_rho + ALIGN_TO_256_BYTES(SEEDBYTES);
    uint8_t *d_m = d_tr + SEEDBYTES;

    // key || mu || w1(packed)
    uint8_t *d_key = d_tr + ALIGN_TO_256_BYTES(SEEDBYTES + mlen);
    uint8_t *d_mu = d_key + SEEDBYTES;
    uint8_t *d_w1_packed = d_mu + CRHBYTES;

    uint8_t *d_rhoprime = d_key + ALIGN_TO_256_BYTES(SEEDBYTES + CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);

    size_t byte_size = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                                           // sig
                       ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES) +                                  // sk
                       ALIGN_TO_256_BYTES(SEEDBYTES) +                                              // d_rho
                       ALIGN_TO_256_BYTES(SEEDBYTES + mlen) +                                       // d_tr || d_m
                       ALIGN_TO_256_BYTES(SEEDBYTES + CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +// d_key || d_mu || d_w1_packed
                       ALIGN_TO_256_BYTES(CRHBYTES);                                                // d_rhoprime

    auto *d_mat = (int32_t *) (d_sign_mem_pool + byte_size);
    int32_t *d_s1 = d_mat + DILITHIUM_K * DILITHIUM_L * DILITHIUM_N;
    int32_t *d_y = d_s1 + DILITHIUM_L * DILITHIUM_N;
    int32_t *d_z = d_y + DILITHIUM_L * DILITHIUM_N;
    int32_t *d_t0 = d_z + DILITHIUM_L * DILITHIUM_N;
    int32_t *d_s2 = d_t0 + DILITHIUM_K * DILITHIUM_N;
    int32_t *d_w1 = d_s2 + DILITHIUM_K * DILITHIUM_N;
    int32_t *d_w0 = d_w1 + DILITHIUM_K * DILITHIUM_N;
    int32_t *d_cp = d_w0 + DILITHIUM_K * DILITHIUM_N;

    //        CUDATimer timer("sign_inner");
    //        timer.start();

    cudaMemcpy2DAsync(d_m, d_sign_mem_pool_pitch, m, m_pitch, mlen, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_sk, d_sign_mem_pool_pitch, sk, CRYPTO_SECRETKEYBYTES, CRYPTO_SECRETKEYBYTES, batch_size, cudaMemcpyHostToDevice, stream);

    cudaMemcpy2DAsync(d_rho, d_sign_mem_pool_pitch, d_sk_rho, d_sign_mem_pool_pitch, SEEDBYTES, batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_tr, d_sign_mem_pool_pitch, d_sk_tr, d_sign_mem_pool_pitch, SEEDBYTES, batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_key, d_sign_mem_pool_pitch, d_sk_key, d_sign_mem_pool_pitch, SEEDBYTES, batch_size, cudaMemcpyDeviceToDevice, stream);

    /* Compute CRH(tr, msg) */
    //    CUDATimer timer_shake_kernel("shake_kernel");
    //    for (size_t i = 0; i < 1000; i++) {
    //        timer_shake_kernel.start();
    //        notmp_shake_kernel<SHAKE256_RATE><<<batch_size, dim3(32, 1), SHAKE256_RATE, stream>>>(
    //                d_mu, d_sign_mem_pool_pitch, CRHBYTES, d_tr, d_sign_mem_pool_pitch, SEEDBYTES + mlen, batch_size);
    notmp_shake_new_kernel<SHAKE256_RATE, 0x1f><<<(batch_size + 3) / 4, dim3(32, 4), SHAKE256_RATE, stream>>>(
            d_mu, d_sign_mem_pool_pitch, CRHBYTES, d_tr, d_sign_mem_pool_pitch, SEEDBYTES + mlen, batch_size);
    //        cudaStreamSynchronize(stream);
    //        timer_shake_kernel.stop();
    //    }

#ifdef DILITHIUM_RANDOMIZED_SIGNING
    uint8_t *h_rhoprime;
    cudaMallocHost(&h_rhoprime, CRHBYTES * batch_size);
    randombytes(h_rhoprime, CRHBYTES * batch_size);
    cudaMemcpy2DAsync(d_rhoprime, d_sign_mem_pool_pitch, h_rhoprime, CRHBYTES, CRHBYTES, batch_size, cudaMemcpyHostToDevice, stream);
    cudaFreeHost(h_rhoprime);
#else
    notmp_shake_new_kernel<SHAKE256_RATE, 0x1f><<<(batch_size + 3) / 4, dim3(32, 4), SHAKE256_RATE, stream>>>(
            d_rhoprime, d_sign_mem_pool_pitch, CRHBYTES, d_key, d_sign_mem_pool_pitch, SEEDBYTES + CRHBYTES, batch_size);
#endif

    //    CUDATimer timer_mat_expand("mat_expand");
    //    timer_mat_expand.start();
    /* Expand matrix and transform vectors */
    //    polyvec_matrix_expand_kernel<<<batch_size, 32, 0, stream>>>(d_mat, d_rho, d_sign_mem_pool_pitch);
    polyvec_matrix_expand_opt_kernel<<<(batch_size + 3) / 4, dim3(32, 4), 0, stream>>>(d_mat, d_rho, d_sign_mem_pool_pitch, batch_size);
    //    timer_mat_expand.stop();

    //    CUDATimer timer_ntt("ntt");
    //    timer_ntt.start();
    //    //    ntt_radix8_kernel<DILITHIUM_L><<<batch_size, 32, 0, stream>>>(d_s1, d_sign_mem_pool_pitch);
    //    ntt_radix2_kernel<DILITHIUM_L><<<batch_size, 128, 0, stream>>>(d_s1, d_sign_mem_pool_pitch);
    //    timer_ntt.stop();

    //    CUDATimer timer_unpack("unpack_fuse_ntt");

    //    timer_unpack.start();
    //    unpack_fuse_ntt_kernel<DILITHIUM_L, POLYETA_PACKEDBYTES><<<batch_size, 32, 0, stream>>>(d_s1, d_sk_s1_packed, d_sign_mem_pool_pitch);
    //    unpack_fuse_ntt_kernel<DILITHIUM_K, POLYETA_PACKEDBYTES><<<batch_size, 32, 0, stream>>>(d_s2, d_sk_s2_packed, d_sign_mem_pool_pitch);
    //    unpack_fuse_ntt_kernel<DILITHIUM_K, POLYT0_PACKEDBYTES><<<batch_size, 32, 0, stream>>>(d_t0, d_sk_t0_packed, d_sign_mem_pool_pitch);

    //    unpack_fuse_ntt_radix2_kernel<<<batch_size, 128, 0, stream>>>(
    //            d_s1, d_s2, d_t0,
    //            d_sk_s1_packed, d_sk_s2_packed, d_sk_t0_packed,
    //            d_sign_mem_pool_pitch);

    unpack_fuse_ntt_radix2_opt_kernel<<<batch_size, 128, 0, stream>>>(
            d_s1, d_s2, d_t0,
            d_sk_s1_packed, d_sk_s2_packed, d_sk_t0_packed,
            d_sign_mem_pool_pitch);
    //    cudaStreamSynchronize(stream);
    //    timer_unpack.stop();

    // temp pool

    uint8_t *d_temp_sig = d_temp_mem_pool + 0;
    uint8_t *d_temp_seed = d_temp_sig;
    uint8_t *d_temp_z_packed = d_temp_seed + SEEDBYTES;
    uint8_t *d_temp_hint = d_temp_z_packed + DILITHIUM_L * POLYZ_PACKEDBYTES;

    uint8_t *d_temp_mu = d_temp_sig + ALIGN_TO_256_BYTES(CRYPTO_BYTES);
    uint8_t *d_temp_w1_packed = d_temp_mu + CRHBYTES;

    byte_size = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                              // sig
                ALIGN_TO_256_BYTES(CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);// d_mu || d_w1_packed

    auto *d_temp_y = (int32_t *) (d_temp_mem_pool + byte_size);
    int32_t *d_temp_z = d_temp_y + DILITHIUM_L * DILITHIUM_N;
    int32_t *d_temp_w0 = d_temp_z + DILITHIUM_L * DILITHIUM_N;
    int32_t *d_temp_w1 = d_temp_w0 + DILITHIUM_K * DILITHIUM_N;
    int32_t *d_temp_cp = d_temp_w1 + DILITHIUM_K * DILITHIUM_N;

    //    CUDATimer timer_cp("compute_cp");
    //    CUDATimer timer_w("compute_w");
    //        CUDATimer timer_y("compute_y");
    //    CUDATimer timer_rej_loop("timer_rej_loop");

    for (auto &i: lut.sign_lut) {
        i.done_flag = 0;
        i.nonce = 0;
    }

rej:
    //    timer_rej_loop.start();

    // repeatedly use sign_lut to fill exec_lut
    size_t exec_idx = 0;
    for (size_t sign_idx = 0; sign_idx < batch_size; sign_idx++) {
        if (lut.sign_lut[sign_idx].done_flag == 0) {
            lut.h_exec_lut[exec_idx++] = (sign_idx << 16) | (lut.sign_lut[sign_idx].nonce++);
            if (exec_idx == exec_threshold) {
                // exec_lut is full
                break;
            }
        }
    }

    if (exec_idx == exec_threshold) {
        // exec_lut is full
        cudaMemcpyAsync(lut.d_exec_lut, lut.h_exec_lut, sizeof(uint32_t) * exec_threshold, cudaMemcpyHostToDevice, stream);

        cudaMemsetAsync(lut.d_done_lut, 0, sizeof(uint8_t) * exec_threshold, stream);

        //        timer_y.start();
        //        compute_y_kernel<<<exec_threshold, 32, 0, stream>>>(
        //                d_y,
        //                d_rhoprime,
        //                lut.d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch);

        compute_y_opt_kernel<<<(exec_threshold + 3) / 4, dim3(32, 4), 0, stream>>>(
                d_y,
                d_rhoprime,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch, exec_threshold);

        //        cudaStreamSynchronize(stream);
        //        timer_y.stop();

        //        timer_w.start();
        //                compute_w_32t_kernel<<<exec_threshold, 32, 0, stream>>>(
        //                        d_z, d_w0, d_w1, d_w1_packed,
        //                        d_y, d_mat,
        //                        d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch);

        compute_w_128t_kernel<<<exec_threshold, 128, 0, stream>>>(
                d_w0, d_w1, d_w1_packed,
                d_y, d_mat,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch);

        //        cudaStreamSynchronize(stream);
        //        timer_w.stop();

        //        timer_cp.start();
        compute_cp_kernel<<<exec_threshold, 32, 0, stream>>>(
                d_cp, d_mu,
                d_seed,
                d_mu,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch);

        //        compute_cp_opt_kernel<<<(exec_threshold + 3) / 4, dim3(32, 4), 0, stream>>>(
        //                d_cp, d_mu,
        //                d_seed,
        //                d_mu,
        //                lut.d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch, exec_threshold);

        //        cudaStreamSynchronize(stream);
        //        timer_cp.stop();

        //        CUDATimer timer_rej("rej");
        //        timer_rej.start();
        //        rej_loop_32t_kernel<<<exec_threshold, 32, 0, stream>>>(
        //                d_y, d_z, d_w0, d_w1, d_cp,
        //                d_z_packed, d_hint,
        //                d_done_lut,
        //                d_s1, d_s2, d_t0,
        //                d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch);

        rej_loop_128t_kernel<<<exec_threshold, 128, 0, stream>>>(
                d_y, d_z, d_w0, d_w1, d_cp,
                d_z_packed, d_hint,
                lut.d_done_lut,
                d_s1, d_s2, d_t0,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_sign_mem_pool_pitch);

        //        cudaStreamSynchronize(stream);
        //        timer_rej.stop();

        cudaMemcpyAsync(lut.h_done_lut, lut.d_done_lut, sizeof(uint8_t) * exec_threshold, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        // update sign_lut
        for (size_t i = 0; i < exec_threshold; i++) {
            if (lut.h_done_lut[i]) {
                // copy result to sign_lut
                size_t sign_idx = lut.h_exec_lut[i] >> 16;
                lut.sign_lut[sign_idx].done_flag = 1;
            }
        }
    } else {
        // exec_lut is not full, need to predict nonce to fulfill it
        do {
            for (size_t sign_idx = 0; sign_idx < batch_size; sign_idx++) {
                if (lut.sign_lut[sign_idx].done_flag == 0) {
                    lut.h_exec_lut[exec_idx++] = (sign_idx << 16) | (lut.sign_lut[sign_idx].nonce++);
                    if (exec_idx == exec_threshold) {
                        // exec_lut is full
                        break;
                    }
                }
            }
        } while (exec_idx < exec_threshold);

        cudaMemset2DAsync(d_temp_hint, d_temp_mem_pool_pitch, 0, POLYVECH_PACKEDBYTES, exec_threshold, stream);

        cudaMemcpyAsync(lut.d_exec_lut, lut.h_exec_lut, sizeof(uint32_t) * exec_threshold, cudaMemcpyHostToDevice, stream);

        cudaMemsetAsync(lut.d_done_lut, 0, sizeof(uint8_t) * exec_threshold, stream);

        //        timer_y.start();
        //        compute_y_kernel<<<exec_threshold, 32, 0, stream>>>(
        //                d_temp_y,
        //                d_rhoprime,
        //                lut.d_exec_lut, d_sign_mem_pool_pitch, d_temp_mem_pool_pitch);

        compute_y_opt_kernel<<<(exec_threshold + 3) / 4, dim3(32, 4), 0, stream>>>(
                d_temp_y,
                d_rhoprime,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_temp_mem_pool_pitch, exec_threshold);

        //        cudaStreamSynchronize(stream);
        //        timer_y.stop();

        //        timer_w.start();
        //        compute_w_32t_kernel<<<exec_threshold, 32, 0, stream>>>(
        //                d_temp_z, d_temp_w0, d_temp_w1, d_temp_w1_packed,
        //                d_temp_y, d_mat,
        //                d_exec_lut, d_sign_mem_pool_pitch, d_temp_mem_pool_pitch);

        compute_w_128t_kernel<<<exec_threshold, 128, 0, stream>>>(
                d_temp_w0, d_temp_w1, d_temp_w1_packed,
                d_temp_y, d_mat,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_temp_mem_pool_pitch);
        //        cudaStreamSynchronize(stream);
        //        timer_w.stop();

        //        timer_cp.start();
        compute_cp_kernel<<<exec_threshold, 32, 0, stream>>>(
                d_temp_cp, d_temp_mu,
                d_temp_seed,
                d_mu,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_temp_mem_pool_pitch);
        //        cudaStreamSynchronize(stream);
        //        timer_cp.stop();

        //        CUDATimer timer_rej("rej");
        //        timer_rej.start();
        //        rej_loop_32t_kernel<<<exec_threshold, 32, 0, stream>>>(
        //                d_temp_y, d_temp_z, d_temp_w0, d_temp_w1, d_temp_cp,
        //                d_temp_z_packed, d_temp_hint,
        //                d_done_lut,
        //                d_s1, d_s2, d_t0,
        //                d_exec_lut, d_sign_mem_pool_pitch, d_temp_mem_pool_pitch);

        rej_loop_128t_kernel<<<exec_threshold, 128, 0, stream>>>(
                d_temp_y, d_temp_z, d_temp_w0, d_temp_w1, d_temp_cp,
                d_temp_z_packed, d_temp_hint,
                lut.d_done_lut,
                d_s1, d_s2, d_t0,
                lut.d_exec_lut, d_sign_mem_pool_pitch, d_temp_mem_pool_pitch);
        //        cudaStreamSynchronize(stream);
        //        timer_rej.stop();

        cudaMemcpyAsync(lut.h_done_lut, lut.d_done_lut, sizeof(uint8_t) * exec_threshold, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        // copy done signatures to output buffer
        size_t copy_count = 0;
        for (exec_idx = 0; exec_idx < exec_threshold; exec_idx++) {
            if (lut.h_done_lut[exec_idx]) {
                // check if this signature has a smaller correct nonce this round
                size_t sign_idx = lut.h_exec_lut[exec_idx] >> 16;
                if (copy_count == 0) {
                    // copy this signature
                    lut.h_copy_lut[copy_count].exec_idx = exec_idx;
                    lut.h_copy_lut[copy_count++].sign_idx = sign_idx;
                    continue;
                } else {
                    size_t loop = copy_count;
                    for (size_t i = 0; i < loop; i++) {
                        if (lut.h_copy_lut[i].sign_idx == sign_idx) {
                            break;
                        }
                        if (i == loop - 1) {
                            // copy this signature
                            lut.h_copy_lut[copy_count].exec_idx = exec_idx;
                            lut.h_copy_lut[copy_count++].sign_idx = sign_idx;
                        }
                    }
                }
            }
        }

        if (copy_count == 0) {
            // no signature is done this round
            goto rej;
        }

        cudaMemcpyAsync(lut.d_copy_lut, lut.h_copy_lut, sizeof(copy_lut_element) * copy_count, cudaMemcpyHostToDevice, stream);

        // launch copy kernel
        sig_copy_kernel<<<copy_count, 96, 0, stream>>>(
                d_sig, d_sign_mem_pool_pitch,
                d_temp_sig, d_temp_mem_pool_pitch,
                lut.d_copy_lut);

        // update sign_lut
        for (size_t copy_idx = 0; copy_idx < copy_count; copy_idx++) {
            lut.sign_lut[lut.h_copy_lut[copy_idx].sign_idx].done_flag = 1;
        }
    }

    //    cudaStreamSynchronize(stream);
    //    timer_rej_loop.stop();

    // check if all done
    for (auto &e: lut.sign_lut) {
        if (e.done_flag == 0) {
            goto rej;
        }
    }

    cudaMemcpy2DAsync(sig, sig_pitch, d_sig, d_sign_mem_pool_pitch, CRYPTO_BYTES, batch_size, cudaMemcpyDeviceToHost, stream);

    *siglen = CRYPTO_BYTES;

    //        cudaStreamSynchronize(stream);
    //        timer.stop();

    return 0;
}

void crypto_sign_verify(int *ret,
                        const uint8_t *sig, size_t sig_pitch, size_t siglen,
                        const uint8_t *m, size_t m_pitch, size_t mlen,
                        const uint8_t *pk,
                        uint8_t *d_verify_mem_pool, size_t verify_mem_pool_pitch,
                        size_t batch_size, cudaStream_t stream) {

    if (siglen != CRYPTO_BYTES) {
        for (size_t i = 0; i < batch_size; ++i) ret[i] = -1;
        return;
    }

    size_t byte_size = ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                               // sig
                       ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +                      // pk (rho || t1(packed))
                       ALIGN_TO_256_BYTES(SEEDBYTES + mlen) +                           // muprime || m
                       ALIGN_TO_256_BYTES(CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +// mu || w1_prime_packed
                       ALIGN_TO_256_BYTES(SEEDBYTES);                                   // c_tilde

    uint8_t *d_sig = d_verify_mem_pool + 0;
    uint8_t *d_c = d_sig + 0;
    uint8_t *d_z_packed = d_c + SEEDBYTES;
    uint8_t *d_h_packed = d_z_packed + DILITHIUM_L * POLYZ_PACKEDBYTES;

    uint8_t *d_pk = d_sig + ALIGN_TO_256_BYTES(CRYPTO_BYTES);
    uint8_t *d_rho = d_pk + 0;
    uint8_t *d_t1_packed = d_rho + SEEDBYTES;

    uint8_t *d_muprime = d_pk + ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES);
    uint8_t *d_m = d_muprime + SEEDBYTES;

    uint8_t *d_mu = d_muprime + ALIGN_TO_256_BYTES(SEEDBYTES + mlen);
    uint8_t *d_w1prime_packed = d_mu + CRHBYTES;

    uint8_t *d_ctilde = d_mu + ALIGN_TO_256_BYTES(CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);

    auto *d_mat = (int32_t *) (d_verify_mem_pool + byte_size);
    int32_t *d_z = d_mat + DILITHIUM_K * DILITHIUM_L * DILITHIUM_N;
    int32_t *d_t1 = d_z + DILITHIUM_L * DILITHIUM_N;
    int32_t *d_w1prime = d_t1 + DILITHIUM_K * DILITHIUM_N;
    int32_t *d_h = d_w1prime + DILITHIUM_K * DILITHIUM_N;
    int32_t *d_cp = d_h + DILITHIUM_K * DILITHIUM_N;
    int *d_ret = d_cp + DILITHIUM_N;

    cudaMemset2DAsync(d_h, verify_mem_pool_pitch, 0, DILITHIUM_K * DILITHIUM_N * sizeof(int32_t), batch_size, stream);
    cudaMemset2DAsync(d_ret, verify_mem_pool_pitch, -1, sizeof(int), batch_size, stream);

    cudaMemcpy2DAsync(d_sig, verify_mem_pool_pitch, sig, sig_pitch, CRYPTO_BYTES, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_m, verify_mem_pool_pitch, m, m_pitch, mlen, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_pk, verify_mem_pool_pitch, pk, CRYPTO_PUBLICKEYBYTES, CRYPTO_PUBLICKEYBYTES, batch_size, cudaMemcpyHostToDevice, stream);

    gpu_verify<<<batch_size, 32, 0, stream>>>(d_ret,
                                              d_c, d_z_packed, d_h_packed,
                                              d_rho, d_t1_packed,
                                              d_muprime, d_m, d_mu, d_w1prime_packed, d_ctilde,
                                              d_mat, d_z, d_t1, d_w1prime, d_h, d_cp,
                                              mlen, verify_mem_pool_pitch);

    cudaMemcpy2DAsync(ret, sizeof(int), d_ret, verify_mem_pool_pitch, sizeof(int), batch_size, cudaMemcpyDeviceToHost, stream);
}

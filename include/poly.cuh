/* Modified form https://github.com/pq-crystals/dilithium.
 * Under Apache 2.0 License.
 */

#pragma once

#include <cstdint>

#include "params.h"

typedef struct {
    int32_t coeffs[DILITHIUM_N];
} poly;

/* Vectors of polynomials of length L */
typedef struct {
    poly vec[DILITHIUM_L];
} polyvecl;

/* Vectors of polynomials of length K */
typedef struct {
    poly vec[DILITHIUM_K];
} polyveck;

/**
 * check is norm
 * @tparam B
 * @param a
 * @return True/False
 */
template<int32_t B>
__device__ int chknorm(const int32_t a) {
    unsigned int i;
    int32_t t;

    if (B > (DILITHIUM_Q - 1) / 8)
        return 1;

    /* It is ok to leak which coefficient violates the bound since
     * the probability for each coefficient is independent of secret
     * data, but we must not leak the sign of the centralized representative.
     */
    for (i = 0; i < DILITHIUM_N; ++i) {
        /* Absolute value */
        t = a >> 31;
        t = a - (t & 2 * a);

        if (t >= B) {
            return 1;
        }
    }

    return 0;
}

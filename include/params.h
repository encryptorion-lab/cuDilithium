/* Modified form https://github.com/pq-crystals/dilithium.
 * Under Apache 2.0 License.
 */

#pragma once

#define SEEDBYTES 32
#define CRHBYTES 64
#define DILITHIUM_N 256
#define DILITHIUM_Q 8380417
#define DILITHIUM_D 13
#define ROOT_OF_UNITY 1753

#if DILITHIUM_MODE == 2
#define DILITHIUM_K 4
#define DILITHIUM_L 4
#define ETA 2
#define TAU 39
#define BETA 78
#define GAMMA1 (1 << 17)
#define GAMMA2 ((DILITHIUM_Q - 1) / 88)
#define OMEGA 80
#define HINT_PAD_SIZE 4

#elif DILITHIUM_MODE == 3
#define DILITHIUM_K 6
#define DILITHIUM_L 5
#define ETA 4
#define TAU 49
#define BETA 196
#define GAMMA1 (1 << 19)
#define GAMMA2 ((DILITHIUM_Q - 1) / 32)
#define OMEGA 55
#define HINT_PAD_SIZE 3

#elif DILITHIUM_MODE == 5
#define DILITHIUM_K 8
#define DILITHIUM_L 7
#define ETA 2
#define TAU 60
#define BETA 120
#define GAMMA1 (1 << 19)
#define GAMMA2 ((DILITHIUM_Q - 1) / 32)
#define OMEGA 75
#define HINT_PAD_SIZE 5

#endif

#define POLYT1_PACKEDBYTES 320
#define POLYT0_PACKEDBYTES 416
#define POLYVECH_PACKEDBYTES (OMEGA + DILITHIUM_K)

#if GAMMA1 == (1 << 17)
#define POLYZ_PACKEDBYTES 576
#elif GAMMA1 == (1 << 19)
#define POLYZ_PACKEDBYTES 640
#endif

#if GAMMA2 == (DILITHIUM_Q - 1) / 88
#define POLYW1_PACKEDBYTES 192
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
#define POLYW1_PACKEDBYTES 128
#endif

#if ETA == 2
#define POLYETA_PACKEDBYTES 96
#elif ETA == 4
#define POLYETA_PACKEDBYTES 128
#endif

#define CRYPTO_PUBLICKEYBYTES (SEEDBYTES + DILITHIUM_K * POLYT1_PACKEDBYTES)
#define CRYPTO_SECRETKEYBYTES (3 * SEEDBYTES + DILITHIUM_L * POLYETA_PACKEDBYTES + DILITHIUM_K * POLYETA_PACKEDBYTES + DILITHIUM_K * POLYT0_PACKEDBYTES)
#define CRYPTO_BYTES (SEEDBYTES + DILITHIUM_L * POLYZ_PACKEDBYTES + POLYVECH_PACKEDBYTES)

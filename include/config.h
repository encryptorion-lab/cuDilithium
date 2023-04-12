#pragma once
//#define DILITHIUM_MODE 2
//#define DILITHIUM_USE_AES
//#define DILITHIUM_RANDOMIZED_SIGNING
//#define USE_RDPMC
//#define DBENCH

#ifndef DILITHIUM_MODE
#define DILITHIUM_MODE 2
#endif

#if DILITHIUM_MODE == 2
#define CRYPTO_ALGNAME "Dilithium2"
#define DILITHIUM_NAMESPACETOP cudilithium2
#define DILITHIUM_NAMESPACE(s) cudilithium2_##s
#elif DILITHIUM_MODE == 3
#define CRYPTO_ALGNAME "Dilithium3"
#define DILITHIUM_NAMESPACETOP cudilithium3
#define DILITHIUM_NAMESPACE(s) cudilithium3_##s
#elif DILITHIUM_MODE == 5
#define CRYPTO_ALGNAME "Dilithium5"
#define DILITHIUM_NAMESPACETOP cudilithium5
#define DILITHIUM_NAMESPACE(s) cudilithium5_##s
#endif

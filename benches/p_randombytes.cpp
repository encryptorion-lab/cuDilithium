#include <stdint.h>
#include <stdio.h>

#include "fips202.h"

void randombytes(uint8_t *out, size_t outlen) {
    unsigned int i;
    uint8_t buf[8];
    static uint64_t ctr = 0;

    for (i = 0; i < 8; ++i)
        buf[i] = ctr >> 8 * i;

    ctr++;
    shake128(out, outlen, buf, 8);
}

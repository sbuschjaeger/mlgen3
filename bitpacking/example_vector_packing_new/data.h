#ifndef __ARRAY_INT_8_1_32_1_H__
#define __ARRAY_INT_8_1_32_1_H__

#include <stdint.h>

#define M_d 1
#define K_d 32
#define N_d 1

const int8_t zp_input = 0;
const int8_t zp_weights = 0;

int8_t input[32] = {43, 99, 75, -20, -87, 48, 50, 107, 55, 112, 72, 56, 12, 50, 7, 67, -128, 49, 31, 44, -88, -58, 89, 21, -25, -78, -98, 88, -3, 103, -59, -109};

int8_t weights[32] = {-53, -7, 57, 96, 104, -60, 52, -55, 101, 100, -88, -116, 8, -53, -125, 114,-53, -7, 57, 96, 104, -60, 52, -55, 101, 100, -88, -116, 8, -53, -125, 114};

int32_t bias[N_d] = {
    0
};

int32_t golden[M_d * N_d] = {
    -19908
};

#endif  // __ARRAY_INT_8_1_32_1_H__
#ifndef __ARRAY_INT_8_{{M}}_{{K}}_{{N}}_H__
#define __ARRAY_INT_8_{{M}}_{{K}}_{{N}}_H__

#include <stdint.h>

#define M_d {{M}}
#define K_d {{K}}
#define N_d {{N}}

const int8_t zp_input = 0;
const int8_t zp_weights = 0;

{{INPUT_ARRAY}}

{{WEIGHTS_ARRAY}}

int32_t bias[N_d] = {
    {{BIAS_VALUES}}
};

int32_t golden[M_d * N_d] = {
    {{GOLDEN_VALUES}}
};

#endif  // __ARRAY_INT_8_{{M}}_{{K}}_{{N}}_H__
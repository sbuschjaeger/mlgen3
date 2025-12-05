#ifndef __ARRAY_INT_8_{{M}}_{{K}}_{{N}}_H__
#define __ARRAY_INT_8_{{M}}_{{K}}_{{N}}_H__

#include <stdint.h>

#define M_d {{M}}
#define K_d {{K}}
#define N_d {{N}}

const int8_t zp_input = 0;
const int8_t zp_weights = 0;

{{INPUT_PACKED}}

{{WEIGHTS_PACKED}}

{{INPUT_UNPACKED}}

{{WEIGHTS_UNPACKED}}

int32_t bias[N_d] = {
    {{BIAS_VALUES}}
};

{{GOLDEN_ARRAY}}

#endif  // __ARRAY_INT_8_{{M}}_{{K}}_{{N}}_H__

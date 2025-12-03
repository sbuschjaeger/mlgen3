#ifndef MATQUANT_PT_C_H
#define MATQUANT_PT_C_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Model configuration */
#define NUM_LAYERS 7
#define QUANTIZE_SIGNED 1

/* Function prototypes */
int load_model_parameters(void);
void predict(const float* input, float* output, int input_size, int output_size);

#endif /* MATQUANT_PT_C_H */

#ifndef MQ_PT_MLP_H
#define MQ_PT_MLP_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Model configuration */
#define NUM_LAYERS 5
#define QUANTIZE_SIGNED 1

/* Function prototypes */
int load_model_parameters(void);
void predict(const float* input, float* output, int input_size, int output_size);

#endif /* MQ_PT_MLP_H */

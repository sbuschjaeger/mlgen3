#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matquant_pt_c.h"

/* Read CSV file and return data as dynamically allocated arrays */
int read_csv(const char* path, float*** features_ptr, int** labels_ptr, int* num_samples_ptr, int* num_features_ptr) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", path);
        return 0;
    }
    
    /* Read header to find label position */
    char header[65536];
    if (!fgets(header, sizeof(header), file)) {
        fclose(file);
        return 0;
    }
    
    int label_pos = 0;
    char* token = strtok(header, ",");
    while (token != NULL) {
        if (strstr(token, "label") != NULL) break;
        label_pos++;
        token = strtok(NULL, ",");
    }
    
    /* Count lines and features */
    int num_samples = 0;
    int num_features = 0;
    rewind(file);
    
    /* Skip header line */
    if (fgets(header, sizeof(header), file) == NULL) {
        fclose(file);
        return 0;
    }
    
    while (fgets(header, sizeof(header), file)) {
        if (num_samples == 0) {
            /* Count features from first line */
            char line_copy[65536];
            strncpy(line_copy, header, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';
            
            char* tok = strtok(line_copy, ",");
            while (tok != NULL) {
                num_features++;
                tok = strtok(NULL, ",");
            }
            num_features--; /* Subtract label column */
        }
        num_samples++;
    }
    
    /* Allocate memory */
    float** features = (float**)malloc(num_samples * sizeof(float*));
    int* labels = (int*)malloc(num_samples * sizeof(int));
    
    if (!features || !labels) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        free(features);
        free(labels);
        return 0;
    }
    
    /* Read data */
    rewind(file);
    if (fgets(header, sizeof(header), file) == NULL) { /* Skip header */
        fclose(file);
        free(features);
        free(labels);
        return 0;
    }
    
    for (int i = 0; i < num_samples; i++) {
        features[i] = (float*)malloc(num_features * sizeof(float));
        if (!features[i]) {
            fprintf(stderr, "Error: Memory allocation failed for sample %d\n", i);
            fclose(file);
            /* Free already allocated memory */
            for (int j = 0; j < i; j++) {
                free(features[j]);
            }
            free(features);
            free(labels);
            return 0;
        }
        
        if (!fgets(header, sizeof(header), file)) break;
        
        int col = 0;
        int feat_idx = 0;
        token = strtok(header, ",");
        while (token != NULL) {
            if (col == label_pos) {
                labels[i] = atoi(token);
            } else {
                features[i][feat_idx++] = (float)atof(token);
            }
            col++;
            token = strtok(NULL, ",");
        }
    }
    
    fclose(file);
    
    /* Set output parameters */
    *features_ptr = features;
    *labels_ptr = labels;
    *num_samples_ptr = num_samples;
    *num_features_ptr = num_features;
    
    return 1;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <csv_path> <repetitions> [seed]\n", argv[0]);
        return 1;
    }
    
    const char* path = argv[1];
    int repeat = atoi(argv[2]);
    unsigned int seed = (argc > 3) ? (unsigned int)atoi(argv[3]) : 707;
    
    srand(seed);
    printf("Random seed set to: %u\n", seed);
    
    /* Load dataset */
    float** features = NULL;
    int* labels = NULL;
    int num_samples = 0;
    int num_features = 0;
    
    if (!read_csv(path, &features, &labels, &num_samples, &num_features)) {
        fprintf(stderr, "Error: Failed to load dataset\n");
        return 1;
    }
    
    printf("Loaded %d samples with %d features\n", num_samples, num_features);
    
    /* Allocate output buffer */
    float* output = (float*)malloc(10 * sizeof(float));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed for output\n");
        /* Free allocated memory */
        for (int i = 0; i < num_samples; i++) {
            free(features[i]);
        }
        free(features);
        free(labels);
        return 1;
    }
    
    /* Benchmark */
    int matches = 0;
    clock_t start = clock();
    
    for (int r = 0; r < repeat; r++) {
        matches = 0;
        for (int i = 0; i < num_samples; i++) {
            predict(features[i], output, num_features, 10);
            
            /* Find argmax */
            int argmax = 0;
            float max_val = output[0];
            for (int j = 1; j < 10; j++) {
                if (output[j] > max_val) {
                    max_val = output[j];
                    argmax = j;
                }
            }
            
            if (argmax == labels[i]) {
                matches++;
            }
        }
    }
    
    clock_t end = clock();
    double runtime = ((double)(end - start)) / CLOCKS_PER_SEC / (num_samples * repeat) * 1000.0;
    float accuracy = (float)matches / num_samples * 100.0f;
    
    printf("RUNNING BENCHMARK WITH %d REPETITIONS\n", repeat);
    printf("Accuracy: %.2f %%\n", accuracy);
    printf("Latency: %.6f [ms/elem]\n", runtime);
    
    /* Cleanup */
    free(output);
    for (int i = 0; i < num_samples; i++) {
        free(features[i]);
    }
    free(features);
    free(labels);
    
    return 0;
}

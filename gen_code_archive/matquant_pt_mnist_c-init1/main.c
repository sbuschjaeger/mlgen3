#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matquant_pt_c.h"

typedef struct {
    float** features;
    int* labels;
    int num_samples;
    int num_features;
} Dataset;

Dataset* read_csv(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", path);
        return NULL;
    }
    
    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    
    /* Read header to find label position */
    char header[65536];
    if (!fgets(header, sizeof(header), file)) {
        fclose(file);
        free(data);
        return NULL;
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
        free(data);
        return NULL;
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
    
    data->num_samples = num_samples;
    data->num_features = num_features;
    data->features = (float**)malloc(num_samples * sizeof(float*));
    data->labels = (int*)malloc(num_samples * sizeof(int));
    
    /* Read data */
    rewind(file);
    if (fgets(header, sizeof(header), file) == NULL) { /* Skip header */
        fclose(file);
        for (int i = 0; i < num_samples; i++) {
            free(data->features[i]);
        }
        free(data->features);
        free(data->labels);
        free(data);
        return NULL;
    }
    
    for (int i = 0; i < num_samples; i++) {
        data->features[i] = (float*)malloc(num_features * sizeof(float));
        if (!fgets(header, sizeof(header), file)) break;
        
        int col = 0;
        int feat_idx = 0;
        token = strtok(header, ",");
        while (token != NULL) {
            if (col == label_pos) {
                data->labels[i] = atoi(token);
            } else {
                data->features[i][feat_idx++] = (float)atof(token);
            }
            col++;
            token = strtok(NULL, ",");
        }
    }
    
    fclose(file);
    return data;
}

void free_dataset(Dataset* data) {
    if (!data) return;
    for (int i = 0; i < data->num_samples; i++) {
        free(data->features[i]);
    }
    free(data->features);
    free(data->labels);
    free(data);
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
    Dataset* data = read_csv(path);
    if (!data) {
        fprintf(stderr, "Error: Failed to load dataset\n");
        return 1;
    }
    
    printf("Loaded %d samples with %d features\n", data->num_samples, data->num_features);
    
    /* Allocate output buffer */
    float* output = (float*)malloc(10 * sizeof(float)); /* Assuming 10 classes */
    
    /* Benchmark */
    int matches = 0;
    clock_t start = clock();
    
    for (int r = 0; r < repeat; r++) {
        matches = 0;
        for (int i = 0; i < data->num_samples; i++) {
            predict(data->features[i], output, data->num_features, 10);
            
            /* Find argmax */
            int argmax = 0;
            float max_val = output[0];
            for (int j = 1; j < 10; j++) {
                if (output[j] > max_val) {
                    max_val = output[j];
                    argmax = j;
                }
            }
            
            if (argmax == data->labels[i]) {
                matches++;
            }
        }
    }
    
    clock_t end = clock();
    double runtime = ((double)(end - start)) / CLOCKS_PER_SEC / (data->num_samples * repeat) * 1000.0;
    float accuracy = (float)matches / data->num_samples * 100.0f;
    
    printf("RUNNING BENCHMARK WITH %d REPETITIONS\n", repeat);
    printf("Accuracy: %.2f %%\n", accuracy);
    printf("Latency: %.6f [ms/elem]\n", runtime);
    
    /* Cleanup */
    free(output);
    free_dataset(data);
    
    return 0;
}

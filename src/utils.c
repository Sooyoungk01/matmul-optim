#include "utils.h"

float max_difference(float *C, float *C_ref, int M, int K) {
    float max_diff = 0;
    for (int i = 0; i < M * K; ++i) {
        float diff = C[i] - C_ref[i];
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

void init_zero(float *A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = 0;
    }
}

void init_rand(float *A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = (float)rand() / RAND_MAX;
    }
}
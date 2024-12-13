#include "matmul.h"

void sgemm_openblas(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
}

void sgemm_naive(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    int i, j, l;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            float prod = 0;
            for (l = 0; l < k; ++l) {
                prod += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = alpha * prod + beta * C[i * n + j];
        }
    }
}

// TODO: add your custom sgemm functions!



void sgemm(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    sgemm_naive(m, n, k, alpha, A, B, beta, C); // TODO: replace this with your implementation!
}
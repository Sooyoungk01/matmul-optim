#include <cblas.h>

void sgemm(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C);
void sgemm_openblas(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C);
void sgemm_naive(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C);
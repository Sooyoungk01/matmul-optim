#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matmul.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s M N K\n", argv[0]);
        return 1;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);

    printf("[Input]\n");
    printf("M = %d, N = %d, K = %d\n\n", M, N, K);

    float *A = (float *)malloc(M * N * sizeof(float));
    float *B = (float *)malloc(N * K * sizeof(float));
    float *C = (float *)malloc(M * K * sizeof(float));
    float *C_ref = (float *)malloc(M * K * sizeof(float));


    /* Correctness check */

    printf("[Correctness check]\n");
    const int repeat_correctness = 1;
    const float epsilon = 1e-5;

    for (int i = 0; i < repeat_correctness; ++i) {
        init_rand(A, M * N);
        init_rand(B, N * K);
        init_zero(C, M * K);

        sgemm_openblas(M, K, N, 1.0, A, B, 0.0, C_ref);
        sgemm(M, K, N, 1.0, A, B, 0.0, C);

        float max_diff = max_difference(C, C_ref, M, K);
        printf("  (%02d, %c) Max difference: %f\n", i, max_diff > epsilon ? 'X' : 'O', max_diff);
    }

    printf("\n");


    /* Performance measurement */

    printf("[Performance measurement]\n");
    const int repeat_performance = 5;

    // OPENBLAS
    printf("OpenBLAS:\n");
    double elapsed_time_ref = 0;
    
    // warm up
    sgemm_openblas(M, K, N, 1.0, A, B, 0.0, C_ref);

    // measure
    for (int i = 0; i < repeat_performance; ++i) {
        init_rand(A, M * N);
        init_rand(B, N * K);
        init_zero(C_ref, M * K);

        const double t0 = clock();
        sgemm_openblas(M, K, N, 1.0, A, B, 0.0, C_ref);
        const double t1 = clock();

        elapsed_time_ref += (t1 - t0) / CLOCKS_PER_SEC;

        printf("  (%02d) Elapsed time: %f (ms)\n", i, (t1 - t0) / CLOCKS_PER_SEC * 1000);
    }
    printf("  > Average elapsed time: %f (ms)\n\n", elapsed_time_ref / repeat_performance * 1000);

    // CUSTOM
    printf("Custom:\n");
    double elapsed_time = 0;

    // warm up
    sgemm(M, K, N, 1.0, A, B, 0.0, C);

    // measure
    for (int i = 0; i < repeat_performance; ++i) {
        init_rand(A, M * N);
        init_rand(B, N * K);
        init_zero(C, M * K);

        const double t0 = clock();
        sgemm(M, K, N, 1.0, A, B, 0.0, C);
        const double t1 = clock();

        elapsed_time += (t1 - t0) / CLOCKS_PER_SEC;

        printf("  (%02d) Elapsed time: %f (ms)\n", i, (t1 - t0) / CLOCKS_PER_SEC * 1000);
    }
    printf("  > Average elapsed time: %f (ms)\n\n", elapsed_time / repeat_performance * 1000);



   // NAIVE
    printf("NAIVE:\n");
    double elapsed_time_naive = 0;

    // warm up
    sgemm_naive(M, K, N, 1.0, A, B, 0.0, C);

    // measure
    for (int i = 0; i < repeat_performance; ++i) {
        init_rand(A, M * N);
        init_rand(B, N * K);
        init_zero(C, M * K);

        const double t0 = clock();
        sgemm_naive(M, K, N, 1.0, A, B, 0.0, C);
        const double t1 = clock();

        elapsed_time_naive += (t1 - t0) / CLOCKS_PER_SEC;

        printf("  (%02d) Elapsed time: %f (ms)\n", i, (t1 - t0) / CLOCKS_PER_SEC * 1000);
    }
    printf("  > Average elapsed time: %f (ms)\n\n", elapsed_time_naive / repeat_performance * 1000);



    printf("Speedup: %f\n\n", elapsed_time_ref / elapsed_time);
    if (elapsed_time / repeat_performance < elapsed_time_ref / repeat_performance) {
        printf("Congratulations! Custom implementation is faster than OpenBLAS.\n");
    } else {
        printf("Cheer up! Custom implementation is slower than OpenBLAS.\n");
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}
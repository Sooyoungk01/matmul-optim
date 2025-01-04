#include <arm_neon.h>
#include <stdio.h>

#define MC 256
#define NC 256
#define KC 256

#define MR 4
#define NR 4

/* buffer */
float _A[MC*KC] __attribute__ ((aligned (16)));
float _B[KC*NC] __attribute__ ((aligned (16)));


void pack_MRxKC(int incRowA, int incColA, float* A, float* _A){
    int i, j;
    for(i=0; i<MR; i++){
        for(j=0; j<KC; j++){
            _A[j*MR+i] = A[i*incRowA + j*incColA];
        }
    }
}

void pack_A(int incRowA, int incColA, float* A, float* _A){
    int i;
    int m = MC/MR;
    for(i=0; i<m; i++){
        pack_MRxKC(incRowA, incColA, A, _A);
        A += MR * incRowA;
        _A += MR * KC;
    }

}

void pack_KCxNR(int incRowB, int incColB, float* B, float* _B){
    int i, j;
    for(i=0; i<KC; i++){
        for(j=0; j<NR; j++){
            _B[j+i*NR] = B[i*incRowB + j*incColB];
        }
    }
}

void pack_B(int incRowB, int incColB, float* B, float* _B){
    int i;
    int n = NC/NR;
    for(i=0; i<n; i++){
        pack_KCxNR(incRowB, incColB, B, _B);
        B += NR * incColB;
        _B += NR * KC;
    }
}

void micro_kernel(float *A, float *B, int incRowC, int incColC, float* C, float alpha, float beta){

    float _AB[MR*NR] __attribute__ ((aligned (16))) = {0};
    float* AB = (float *)__builtin_assume_aligned(_AB, 16);
    int i, j;
    int k = KC;

    __asm__ volatile (
        "ldr    x0, %[A]  \n\t"   // load A into x0
        "ldr    x1, %[B] \n\t"   // load B into x1
        "ldr    x2, %[AB]  \n\t" // load AB into x2
        "mov    x3, %[k]    \n\t" 
        "ld1    {v0.4s}, [x1] \n\t" // btmp = vld1q_f32(B);
        "ld1    {v1.4s}, [x0] \n\t" // atmp0 = vld1q_f32(A);
        "eor    v10.16b, v10.16b, v10.16b \n\t" // ab_00 = vdupq_n_f32(0.0f);
        "eor    v11.16b, v11.16b, v11.16b \n\t" // ab_30 = vdupq_n_f32(0.0f);
        "eor    v12.16b, v12.16b, v12.16b \n\t" // ab_20 = vdupq_n_f32(0.0f);
        "eor    v13.16b, v13.16b, v13.16b \n\t" // ab_10 = vdupq_n_f32(0.0f);
        "                                 \n\t"
            ".DLOOP%=:          \n\t"
            "                   \n\t"
            "ext    v2.16b, v1.16b, v1.16b, 12  \n\t" // atmp3 = vextq_f32(atmp0, atmp0, 3);
            "ext    v3.16b, v1.16b, v1.16b, 8   \n\t" // atmp2 = vextq_f32(atmp0, atmp0, 2);
            "ext    v4.16b, v1.16b, v1.16b, 4   \n\t" // atmp1 = vextq_f32(atmp0, atmp0, 1);
            "                                   \n\t"
            "fmla   v10.4s, v1.4s,  v0.4s       \n\t" // atmp0 = vmulq_f32(atmp0, btmp); ab_00 = vaddq_f32(ab_00, atmp0);
            "                                   \n\t"
            "add    x0, x0, 16                  \n\t" // A += 4;
            "ld1    {v1.4s}, [x0]               \n\t" // atmp0 = vld1q_f32(A+4);
            "                                   \n\t"
            "fmla   v11.4s, v2.4s, v0.4s        \n\t" // atmp3 = vmulq_f32(atmp3, btmp); ab_30 = vaddq_f32(ab_30, atmp3);
            "fmla   v12.4s, v3.4s, v0.4s       \n\t"
            "fmla   v13.4s, v4.4s,  v0.4s       \n\t"
            "                                   \n\t"
            "add    x1, x1, 16                  \n\t" // B += 4;
            "ld1    {v0.4s}, [x1]               \n\t" // btmp = vld1q_f32(B);
            "                                   \n\t"
            "subs   x3,  x3,    1               \n\t" // k--
            "bne    .DLOOP%=                    \n\t"
            "                                   \n\t"
        "st1    {v10.s}[0], [x2]        \n\t" // vst1q_lane(AB[0], ab_00, 0);
        "add    x2, x2, 4              \n\t"
        "st1    {v11.s}[1], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v12.s}[2], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v13.s}[3], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v13.s}[0], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v10.s}[1], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v11.s}[2], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v12.s}[3], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v12.s}[0], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v13.s}[1], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v10.s}[2], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v11.s}[3], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v11.s}[0], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v12.s}[1], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v13.s}[2], [x2]        \n\t"
        "add    x2, x2, 4              \n\t"
        "st1    {v10.s}[3], [x2]        \n\t"
        : // output
        : // input
            [A ]"m" (A),      // 0
            [B] "m" (B),      // 1
            [AB] "m" (AB),     // 2
            [k] "r" (k)       // 3
        : // register clobber list
            "memory", "x0", "x1", "x2", "x3",
            "v0", "v1", "v2", "v3", "v4",
            "v10", "v11", "v12", "v13"
    );


    if(beta == 1){
        for(i=0; i<MR; i++){
            for(j=0; j<NR; j++){
                C[i*incRowC + j] += AB[i*NR+j]*alpha;
            }
        }
    }else{
        for(i=0; i<MR; i++){
            for(j=0; j<NR; j++){
                C[i*incRowC + j] *= beta;
                C[i*incRowC + j] += AB[i*NR+j]*alpha;
            }
        }
    }
}

void macro_kernel(int incRowC, int incColC, float* C, float alpha, float beta){
    int i, j;
    int m = MC/MR;
    int n = NC/NR;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            micro_kernel(&_A[i*MR*KC], &_B[j*NR*KC], incRowC, incColC, &C[i*MR*incRowC + j*NR*incColC], alpha, beta);
        }
    }
}

void sgemm_custom(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C){
    int incRowA = k;
    int incColA = 1;
    int incRowB = n;
    int incColB = 1;
    int incRowC = n;
    int incColC = 1;

    int mt = m / MC;
    int nt = n / NC;
    int kt = k / KC;

    int i, j, l;
    float _beta = beta;

    if (alpha == 0){
        for(i=0; i<m*n; i++){
            C[i] = beta * C[i];
        }
        return;
    }

    for(l=0; l<kt; l++){
        for(i=0; i<mt; i++){
            pack_A(incRowA, incColA, &A[i*MC*incRowA + l*KC*incColA], _A);

            for(j=0; j<nt; j++){
                pack_B(incRowB, incColB, &B[l*KC*incRowB + j*NC*incColB], _B);
                macro_kernel(incRowC, incColC, &C[i*MC*incRowC + j*NC*incColC], alpha, _beta);
            }
        }
        _beta = 1;
    }
}
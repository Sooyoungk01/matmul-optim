#include <arm_neon.h>

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

    float AB[MR*NR] __attribute__ ((aligned (16))) = {0};

    float32x4_t ab_0, ab_1, ab_2, ab_3;

    float32x4_t a;
    float32x4_t b_0, b_1, b_2, b_3;
    float32x4_t tmp;

    int i, j, k;

    ab_0 = vdupq_n_f32(0.0f);
    ab_1 = vdupq_n_f32(0.0f);
    ab_2 = vdupq_n_f32(0.0f);
    ab_3 = vdupq_n_f32(0.0f);
    
    for(k=0; k<KC; k++){
        for(i=0; i<MR; i++){
            for(j=0; j<NR; j++){
                AB[i*NR+j] += A[i+k*MR] * B[j+k*NR];
            }
        }
    }

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
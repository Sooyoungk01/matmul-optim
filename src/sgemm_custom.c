#include <arm_neon.h>
#include <stdio.h>

#define MC 256
#define NC 256
#define KC 256

#define MR 4
#define NR 4

/* buffer */
float _A[MC*KC] __attribute__ ((aligned (8)));
float _B[KC*NC] __attribute__ ((aligned (8)));


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

    float AB[MR*NR] __attribute__ ((aligned (8))) = {0};
    int i, j, k;

    float32x2_t ab_00_11, ab_01_10, ab_02_13, ab_03_12, ab_20_31, ab_21_30, ab_22_33, ab_23_32;
    float32x2_t btmp0, btmp1, atmp0, atmp1;
    float32x2_t atmp0_r, atmp1_r, temp;

    btmp0 = vld1_f32(B);
    btmp1 = vld1_f32(B+2);
    atmp0 = vld1_f32(A);

    ab_00_11 = vdup_n_f32(0.0f);
    ab_01_10 = vdup_n_f32(0.0f);
    ab_02_13 = vdup_n_f32(0.0f);
    ab_03_12 = vdup_n_f32(0.0f);    
    ab_20_31 = vdup_n_f32(0.0f);
    ab_21_30 = vdup_n_f32(0.0f);
    ab_22_33 = vdup_n_f32(0.0f);
    ab_23_32 = vdup_n_f32(0.0f);
    

    for (k=0; k<KC; k++){
        atmp1 = vld1_f32(A+2);

        atmp0_r = vrev64_f32(atmp0);
        atmp1_r = vrev64_f32(atmp1);

        temp = atmp0;
        atmp0 = vmul_f32(atmp0, btmp0);
        temp = vmul_f32(temp, btmp1);
        ab_00_11 = vadd_f32(ab_00_11, atmp0);
        ab_02_13 = vadd_f32(ab_02_13, temp);

        temp = atmp0_r;
        atmp0_r = vmul_f32(atmp0_r, btmp0);
        temp = vmul_f32(temp, btmp1);
        ab_01_10 = vadd_f32(ab_01_10, atmp0_r);
        ab_03_12 = vadd_f32(ab_03_12, temp);

        atmp0 = vld1_f32(A+4); // load in advance

        temp = atmp1;
        atmp1 = vmul_f32(atmp1, btmp0);
        temp = vmul_f32(temp, btmp1);
        ab_20_31 = vadd_f32(ab_20_31, atmp1);
        ab_22_33 = vadd_f32(ab_22_33, temp);

        temp = atmp1_r;
        atmp1_r = vmul_f32(atmp1_r, btmp0);
        temp = vmul_f32(temp, btmp1);
        ab_21_30 = vadd_f32(ab_21_30, atmp1_r);
        ab_23_32 = vadd_f32(ab_23_32, temp);

        btmp0 = vld1_f32(B+4); // load in advance
        btmp1 = vld1_f32(B+6);

        A += 4;
        B += 4;
    }

    AB[0] = vget_lane_f32(ab_00_11, 0);
    AB[1] = vget_lane_f32(ab_01_10, 1);
    AB[2] = vget_lane_f32(ab_02_13, 0);
    AB[3] = vget_lane_f32(ab_03_12, 1);
    AB[4] = vget_lane_f32(ab_01_10, 0);
    AB[5] = vget_lane_f32(ab_00_11, 1);
    AB[6] = vget_lane_f32(ab_03_12, 0);
    AB[7] = vget_lane_f32(ab_02_13, 1);
    AB[8] = vget_lane_f32(ab_20_31, 0);
    AB[9] = vget_lane_f32(ab_21_30, 1);
    AB[10] = vget_lane_f32(ab_22_33, 0);
    AB[11] = vget_lane_f32(ab_23_32, 1);
    AB[12] = vget_lane_f32(ab_21_30, 0);
    AB[13] = vget_lane_f32(ab_20_31, 1);
    AB[14] = vget_lane_f32(ab_23_32, 0);
    AB[15] = vget_lane_f32(ab_22_33, 1);


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
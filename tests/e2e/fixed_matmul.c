#include <math.h>
#include <stdio.h>

extern void fixed_matmul(float *C, float *A, float *B);

int main() {
    float A[256] = {0}, B[256] = {0}, C[256] = {0};
    A[0 * 16 + 0] = 1;
    A[0 * 16 + 1] = 2;
    A[1 * 16 + 0] = 3;
    A[1 * 16 + 1] = 4;
    B[0 * 16 + 0] = 5;
    B[0 * 16 + 1] = 6;
    B[1 * 16 + 0] = 7;
    B[1 * 16 + 1] = 8;
    fixed_matmul(C, A, B);
    printf("C[0,0]=%.0f C[0,1]=%.0f C[1,0]=%.0f C[1,1]=%.0f\n", C[0 * 16 + 0], C[0 * 16 + 1], C[1 * 16 + 0], C[1 * 16 + 1]);
    return 0;
}

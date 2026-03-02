#include <stdio.h>

extern void negate_float(float *out, float *a);
extern void negate_int(int *out, int *a);

int main() {
    float fa = 42.0f, fout = 0.0f;
    negate_float(&fout, &fa);
    printf("%.1f\n", fout);

    int ia = 7, iout = 0;
    negate_int(&iout, &ia);
    printf("%d\n", iout);

    return 0;
}

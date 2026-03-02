#include <stdio.h>

extern void float_arithmetic(float *out, float *a, float *b);

int main() {
    float a = 10.0f, b = 3.0f, out = 0.0f;
    float_arithmetic(&out, &a, &b);
    printf("%.6f\n", out);
    return 0;
}

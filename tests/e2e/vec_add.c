#include <stdio.h>

extern void vec_add(long N, float *out, float *a, float *b);

int main() {
    float a[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float out[5] = {0};
    vec_add(5, out, a, b);
    printf("%.0f %.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3], out[4]);
    return 0;
}

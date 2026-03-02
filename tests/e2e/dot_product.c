#include <stdio.h>

extern void dot_product(long N, float *out, float *a, float *b);

int main() {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[1] = {0.0f};
    dot_product(4, out, a, b);
    printf("%.1f\n", out[0]);
    return 0;
}

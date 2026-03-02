#include <stdio.h>

extern void if_else(float *out, long a, long b);

int main() {
    float out = 0.0f;

    if_else(&out, 1, 5);
    printf("%.1f\n", out);

    if_else(&out, 5, 1);
    printf("%.1f\n", out);

    return 0;
}

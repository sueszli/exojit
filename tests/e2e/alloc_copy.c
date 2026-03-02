#include <math.h>
#include <stdio.h>

extern void alloc_copy(long N, float *x);

int main() {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alloc_copy(4, x);
    int ok = (fabsf(x[0] - 1.0f) < 1e-5f && fabsf(x[1] - 2.0f) < 1e-5f && fabsf(x[2] - 3.0f) < 1e-5f && fabsf(x[3] - 4.0f) < 1e-5f);
    printf("%s\n", ok ? "OK" : "FAIL");
    return 0;
}

#include <math.h>
#include <stdio.h>

extern void assign_2d(float *dst, float *src);

int main() {
    float src[16], dst[16] = {0};
    for (int i = 0; i < 16; i++)
        src[i] = (float)i;
    assign_2d(dst, src);
    int ok = 1;
    for (int i = 0; i < 16; i++) {
        if (fabsf(dst[i] - src[i]) > 1e-5f) {
            ok = 0;
            break;
        }
    }
    printf("%s\n", ok ? "OK" : "FAIL");
    return 0;
}

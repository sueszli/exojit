#include <stdio.h>

extern void scalar_double(float *x);

int main() {
    float x[1] = {21.0f};
    scalar_double(x);
    printf("%.1f\n", x[0]);
    return 0;
}

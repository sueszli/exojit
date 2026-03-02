#include <stdio.h>

extern void compare_and_branch(float *out, long a, long b);

int main() {
    float out = 0.0f;

    compare_and_branch(&out, 3, 7);
    printf("%.1f\n", out);

    compare_and_branch(&out, 7, 3);
    printf("%.1f\n", out);

    return 0;
}

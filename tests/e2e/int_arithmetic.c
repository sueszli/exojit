#include <stdio.h>

extern void int_arithmetic(int *out, int *a, int *b);

int main() {
    int a = 10, b = 3, out = 0;
    int_arithmetic(&out, &a, &b);
    // last op is a/b = 10/3 = 3
    printf("%d\n", out);
    return 0;
}

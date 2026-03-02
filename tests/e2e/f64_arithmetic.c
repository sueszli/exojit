#include <stdio.h>

extern void f64_arithmetic(double *out, double *a, double *b);

int main() {
    double a = 10.0, b = 3.0, out = 0.0;
    f64_arithmetic(&out, &a, &b);
    printf("%.15f\n", out);
    return 0;
}

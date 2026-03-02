#include <stdio.h>

extern void reduce_float(float *x, float *y);
extern void reduce_int(int *x, int *y);

int main() {
    float xf[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float yf[1] = {0.0f};
    reduce_float(xf, yf);
    printf("%.1f\n", yf[0]);

    int xi[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int yi[1] = {0};
    reduce_int(xi, yi);
    printf("%d\n", yi[0]);

    return 0;
}

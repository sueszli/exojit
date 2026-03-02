#include <stdio.h>
#include <string.h>

extern void i8_copy(char *dst, char *src);

int main() {
    char src[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    char dst[8] = {0};
    i8_copy(dst, src);
    int ok = (memcmp(dst, src, 8) == 0);
    printf("%s\n", ok ? "OK" : "FAIL");
    return 0;
}

#include <cstdlib>
#include "../include/test.h"

int main(int argc, char **argv) {
    testHandle_t handle;
    float * __restrict__ A = (float*) calloc(8, sizeof(float));
    float * __restrict__ B = (float*) calloc(16, sizeof(float));
    float * __restrict__ R = (float*) calloc(8, sizeof(float));


    handle = __dace_init_test();
    __program_test(handle, A, B, R);
    __dace_exit_test(handle);

    free(A);
    free(B);
    free(R);


    return 0;
}

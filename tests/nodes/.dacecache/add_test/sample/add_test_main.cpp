#include <cstdlib>
#include "../include/add_test.h"

int main(int argc, char **argv) {
    add_testHandle_t handle;
    int M = 42;
    int N = 42;
    float * __restrict__ A = (float*) calloc((M * N), sizeof(float));
    float * __restrict__ B = (float*) calloc((M * N), sizeof(float));


    handle = __dace_init_add_test(M, N);
    __program_add_test(handle, A, B, M, N);
    __dace_exit_add_test(handle);

    free(A);
    free(B);


    return 0;
}

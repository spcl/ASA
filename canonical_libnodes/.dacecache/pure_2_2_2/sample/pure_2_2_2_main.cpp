#include <cstdlib>
#include "../include/pure_2_2_2.h"

int main(int argc, char **argv) {
    pure_2_2_2Handle_t handle;
    float * __restrict__ A = (float*) calloc(4, sizeof(float));
    float * __restrict__ B = (float*) calloc(4, sizeof(float));
    float * __restrict__ C = (float*) calloc(4, sizeof(float));


    handle = __dace_init_pure_2_2_2();
    __program_pure_2_2_2(handle, A, B, C);
    __dace_exit_pure_2_2_2(handle);

    free(A);
    free(B);
    free(C);


    return 0;
}

#include <cstdlib>
#include "../include/mv_2_2_2.h"

int main(int argc, char **argv) {
    mv_2_2_2Handle_t handle;
    float * __restrict__ A = (float*) calloc(4, sizeof(float));
    float * __restrict__ B = (float*) calloc(4, sizeof(float));
    float * __restrict__ C = (float*) calloc(4, sizeof(float));


    handle = __dace_init_mv_2_2_2();
    __program_mv_2_2_2(handle, A, B, C);
    __dace_exit_mv_2_2_2(handle);

    free(A);
    free(B);
    free(C);


    return 0;
}

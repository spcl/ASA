#include <cstdlib>
#include "../include/mv_4_4_4.h"

int main(int argc, char **argv) {
    mv_4_4_4Handle_t handle;
    float * __restrict__ A = (float*) calloc(16, sizeof(float));
    float * __restrict__ B = (float*) calloc(16, sizeof(float));
    float * __restrict__ C = (float*) calloc(16, sizeof(float));
    float * __restrict__ D = (float*) calloc(16, sizeof(float));


    handle = __dace_init_mv_4_4_4();
    __program_mv_4_4_4(handle, A, B, C, D);
    __dace_exit_mv_4_4_4(handle);

    free(A);
    free(B);
    free(C);
    free(D);


    return 0;
}

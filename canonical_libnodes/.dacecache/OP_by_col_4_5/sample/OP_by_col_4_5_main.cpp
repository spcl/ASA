#include <cstdlib>
#include "../include/OP_by_col_4_5.h"

int main(int argc, char **argv) {
    OP_by_col_4_5Handle_t handle;
    float * __restrict__ A = (float*) calloc(20, sizeof(float));
    float * __restrict__ u = (float*) calloc(4, sizeof(float));
    float * __restrict__ v = (float*) calloc(5, sizeof(float));


    handle = __dace_init_OP_by_col_4_5();
    __program_OP_by_col_4_5(handle, A, u, v);
    __dace_exit_OP_by_col_4_5(handle);

    free(A);
    free(u);
    free(v);


    return 0;
}

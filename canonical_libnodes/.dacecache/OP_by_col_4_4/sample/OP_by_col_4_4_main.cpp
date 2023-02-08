#include <cstdlib>
#include "../include/OP_by_col_4_4.h"

int main(int argc, char **argv) {
    OP_by_col_4_4Handle_t handle;
    float * __restrict__ A = (float*) calloc(16, sizeof(float));
    float * __restrict__ u = (float*) calloc(4, sizeof(float));
    float * __restrict__ v = (float*) calloc(4, sizeof(float));


    handle = __dace_init_OP_by_col_4_4();
    __program_OP_by_col_4_4(handle, A, u, v);
    __dace_exit_OP_by_col_4_4(handle);

    free(A);
    free(u);
    free(v);


    return 0;
}

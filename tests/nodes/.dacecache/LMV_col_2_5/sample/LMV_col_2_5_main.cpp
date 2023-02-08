#include <cstdlib>
#include "../include/LMV_col_2_5.h"

int main(int argc, char **argv) {
    LMV_col_2_5Handle_t handle;
    float * __restrict__ A = (float*) calloc(10, sizeof(float));
    float * __restrict__ x = (float*) calloc(2, sizeof(float));
    float * __restrict__ y = (float*) calloc(5, sizeof(float));


    handle = __dace_init_LMV_col_2_5();
    __program_LMV_col_2_5(handle, A, x, y);
    __dace_exit_LMV_col_2_5(handle);

    free(A);
    free(x);
    free(y);


    return 0;
}

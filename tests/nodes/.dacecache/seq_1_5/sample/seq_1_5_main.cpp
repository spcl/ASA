#include <cstdlib>
#include "../include/seq_1_5.h"

int main(int argc, char **argv) {
    seq_1_5Handle_t handle;
    float * __restrict__ A = (float*) calloc(5, sizeof(float));
    float * __restrict__ x = (float*) calloc(5, sizeof(float));
    float * __restrict__ y = (float*) calloc(1, sizeof(float));


    handle = __dace_init_seq_1_5();
    __program_seq_1_5(handle, A, x, y);
    __dace_exit_seq_1_5(handle);

    free(A);
    free(x);
    free(y);


    return 0;
}

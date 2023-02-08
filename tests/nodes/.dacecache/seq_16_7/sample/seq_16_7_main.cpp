#include <cstdlib>
#include "../include/seq_16_7.h"

int main(int argc, char **argv) {
    seq_16_7Handle_t handle;
    float * __restrict__ A = (float*) calloc(112, sizeof(float));
    float * __restrict__ x = (float*) calloc(7, sizeof(float));
    float * __restrict__ y = (float*) calloc(16, sizeof(float));


    handle = __dace_init_seq_16_7();
    __program_seq_16_7(handle, A, x, y);
    __dace_exit_seq_16_7(handle);

    free(A);
    free(x);
    free(y);


    return 0;
}

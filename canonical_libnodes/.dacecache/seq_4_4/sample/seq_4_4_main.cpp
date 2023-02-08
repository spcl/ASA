#include <cstdlib>
#include "../include/seq_4_4.h"

int main(int argc, char **argv) {
    seq_4_4Handle_t handle;
    float * __restrict__ A = (float*) calloc(16, sizeof(float));
    float * __restrict__ x = (float*) calloc(4, sizeof(float));
    float * __restrict__ y = (float*) calloc(4, sizeof(float));


    handle = __dace_init_seq_4_4();
    __program_seq_4_4(handle, A, x, y);
    __dace_exit_seq_4_4(handle);

    free(A);
    free(x);
    free(y);


    return 0;
}

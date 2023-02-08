#include <cstdlib>
#include "../include/seq_3_4.h"

int main(int argc, char **argv) {
    seq_3_4Handle_t handle;
    float * __restrict__ A = (float*) calloc(12, sizeof(float));
    float * __restrict__ x = (float*) calloc(4, sizeof(float));
    float * __restrict__ y = (float*) calloc(3, sizeof(float));


    handle = __dace_init_seq_3_4();
    __program_seq_3_4(handle, A, x, y);
    __dace_exit_seq_3_4(handle);

    free(A);
    free(x);
    free(y);


    return 0;
}

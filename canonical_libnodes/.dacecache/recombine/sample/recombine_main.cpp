#include <cstdlib>
#include "../include/recombine.h"

int main(int argc, char **argv) {
    recombineHandle_t handle;
    float * __restrict__ A0 = (float*) calloc(3, sizeof(float));
    float * __restrict__ A1 = (float*) calloc(3, sizeof(float));
    float * __restrict__ A2 = (float*) calloc(3, sizeof(float));
    float * __restrict__ __return = (float*) calloc(9, sizeof(float));


    handle = __dace_init_recombine();
    __program_recombine(handle, A0, A1, A2, __return);
    __dace_exit_recombine(handle);

    free(A0);
    free(A1);
    free(A2);
    free(__return);


    return 0;
}

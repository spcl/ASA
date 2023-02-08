#include <cstdlib>
#include "../include/gather_4_5.h"

int main(int argc, char **argv) {
    gather_4_5Handle_t handle;
    float * __restrict__ inp = (float*) calloc(20, sizeof(float));
    float * __restrict__ outp = (float*) calloc(20, sizeof(float));


    handle = __dace_init_gather_4_5();
    __program_gather_4_5(handle, inp, outp);
    __dace_exit_gather_4_5(handle);

    free(inp);
    free(outp);


    return 0;
}

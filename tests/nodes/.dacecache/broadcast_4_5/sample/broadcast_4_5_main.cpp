#include <cstdlib>
#include "../include/broadcast_4_5.h"

int main(int argc, char **argv) {
    broadcast_4_5Handle_t handle;
    float * __restrict__ inp = (float*) calloc(20, sizeof(float));
    float * __restrict__ outp = (float*) calloc(20, sizeof(float));


    handle = __dace_init_broadcast_4_5();
    __program_broadcast_4_5(handle, inp, outp);
    __dace_exit_broadcast_4_5(handle);

    free(inp);
    free(outp);


    return 0;
}

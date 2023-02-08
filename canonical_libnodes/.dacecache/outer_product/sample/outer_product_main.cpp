#include <cstdlib>
#include "../include/outer_product.h"

int main(int argc, char **argv) {
    outer_productHandle_t handle;
    int M = 42;
    int N = 42;
    float * __restrict__ __return = (float*) calloc((M * N), sizeof(float));
    float * __restrict__ u = (float*) calloc(N, sizeof(float));
    float * __restrict__ v = (float*) calloc(M, sizeof(float));


    handle = __dace_init_outer_product(M, N);
    __program_outer_product(handle, __return, u, v, M, N);
    __dace_exit_outer_product(handle);

    free(__return);
    free(u);
    free(v);


    return 0;
}

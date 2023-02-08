#include <dace/dace.h>
typedef void * outer_productHandle_t;
extern "C" outer_productHandle_t __dace_init_outer_product(int M, int N);
extern "C" void __dace_exit_outer_product(outer_productHandle_t handle);
extern "C" void __program_outer_product(outer_productHandle_t handle, float * __restrict__ __return, float * __restrict__ u, float * __restrict__ v, int M, int N);

#include <dace/dace.h>
typedef void * testHandle_t;
extern "C" testHandle_t __dace_init_test();
extern "C" void __dace_exit_test(testHandle_t handle);
extern "C" void __program_test(testHandle_t handle, float * __restrict__ A, float * __restrict__ B, float * __restrict__ R);

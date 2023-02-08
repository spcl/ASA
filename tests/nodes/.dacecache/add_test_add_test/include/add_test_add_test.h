#include <dace/dace.h>
typedef void * add_test_add_testHandle_t;
extern "C" add_test_add_testHandle_t __dace_init_add_test_add_test(int M, int N);
extern "C" void __dace_exit_add_test_add_test(add_test_add_testHandle_t handle);
extern "C" void __program_add_test_add_test(add_test_add_testHandle_t handle, float * __restrict__ A, float * __restrict__ B, int M, int N);

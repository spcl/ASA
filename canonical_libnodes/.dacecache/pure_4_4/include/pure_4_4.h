#include <dace/dace.h>
typedef void * pure_4_4Handle_t;
extern "C" pure_4_4Handle_t __dace_init_pure_4_4();
extern "C" void __dace_exit_pure_4_4(pure_4_4Handle_t handle);
extern "C" void __program_pure_4_4(pure_4_4Handle_t handle, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y);

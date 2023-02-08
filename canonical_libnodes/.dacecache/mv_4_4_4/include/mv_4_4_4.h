#include <dace/dace.h>
typedef void * mv_4_4_4Handle_t;
extern "C" mv_4_4_4Handle_t __dace_init_mv_4_4_4();
extern "C" void __dace_exit_mv_4_4_4(mv_4_4_4Handle_t handle);
extern "C" void __program_mv_4_4_4(mv_4_4_4Handle_t handle, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, float * __restrict__ D);

#include <dace/dace.h>
typedef void * mv_2_2_2Handle_t;
extern "C" mv_2_2_2Handle_t __dace_init_mv_2_2_2();
extern "C" void __dace_exit_mv_2_2_2(mv_2_2_2Handle_t handle);
extern "C" void __program_mv_2_2_2(mv_2_2_2Handle_t handle, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C);

#include <dace/dace.h>
typedef void * LMV_col_2_5Handle_t;
extern "C" LMV_col_2_5Handle_t __dace_init_LMV_col_2_5();
extern "C" void __dace_exit_LMV_col_2_5(LMV_col_2_5Handle_t handle);
extern "C" void __program_LMV_col_2_5(LMV_col_2_5Handle_t handle, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y);

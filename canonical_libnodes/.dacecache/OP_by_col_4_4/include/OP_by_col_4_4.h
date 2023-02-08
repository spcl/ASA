#include <dace/dace.h>
typedef void * OP_by_col_4_4Handle_t;
extern "C" OP_by_col_4_4Handle_t __dace_init_OP_by_col_4_4();
extern "C" void __dace_exit_OP_by_col_4_4(OP_by_col_4_4Handle_t handle);
extern "C" void __program_OP_by_col_4_4(OP_by_col_4_4Handle_t handle, float * __restrict__ A, float * __restrict__ u, float * __restrict__ v);

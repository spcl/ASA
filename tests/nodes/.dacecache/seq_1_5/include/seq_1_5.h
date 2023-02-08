#include <dace/dace.h>
typedef void * seq_1_5Handle_t;
extern "C" seq_1_5Handle_t __dace_init_seq_1_5();
extern "C" void __dace_exit_seq_1_5(seq_1_5Handle_t handle);
extern "C" void __program_seq_1_5(seq_1_5Handle_t handle, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y);

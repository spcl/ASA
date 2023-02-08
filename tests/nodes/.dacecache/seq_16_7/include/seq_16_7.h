#include <dace/dace.h>
typedef void * seq_16_7Handle_t;
extern "C" seq_16_7Handle_t __dace_init_seq_16_7();
extern "C" void __dace_exit_seq_16_7(seq_16_7Handle_t handle);
extern "C" void __program_seq_16_7(seq_16_7Handle_t handle, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y);

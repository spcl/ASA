#include <dace/dace.h>
typedef void * seq_4_4Handle_t;
extern "C" seq_4_4Handle_t __dace_init_seq_4_4();
extern "C" void __dace_exit_seq_4_4(seq_4_4Handle_t handle);
extern "C" void __program_seq_4_4(seq_4_4Handle_t handle, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y);

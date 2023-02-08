#include <dace/dace.h>
typedef void * broadcast_4_5Handle_t;
extern "C" broadcast_4_5Handle_t __dace_init_broadcast_4_5();
extern "C" void __dace_exit_broadcast_4_5(broadcast_4_5Handle_t handle);
extern "C" void __program_broadcast_4_5(broadcast_4_5Handle_t handle, float * __restrict__ inp, float * __restrict__ outp);

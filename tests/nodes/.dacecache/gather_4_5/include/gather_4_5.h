#include <dace/dace.h>
typedef void * gather_4_5Handle_t;
extern "C" gather_4_5Handle_t __dace_init_gather_4_5();
extern "C" void __dace_exit_gather_4_5(gather_4_5Handle_t handle);
extern "C" void __program_gather_4_5(gather_4_5Handle_t handle, float * __restrict__ inp, float * __restrict__ outp);

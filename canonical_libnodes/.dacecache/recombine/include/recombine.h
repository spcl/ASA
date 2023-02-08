#include <dace/dace.h>
typedef void * recombineHandle_t;
extern "C" recombineHandle_t __dace_init_recombine();
extern "C" void __dace_exit_recombine(recombineHandle_t handle);
extern "C" void __program_recombine(recombineHandle_t handle, float * __restrict__ A0, float * __restrict__ A1, float * __restrict__ A2, float * __restrict__ __return);

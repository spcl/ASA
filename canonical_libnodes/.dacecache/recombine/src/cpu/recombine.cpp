/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct recombine_t {

};

void __program_recombine_internal(recombine_t *__state, float * __restrict__ A0, float * __restrict__ A1, float * __restrict__ A2, float * __restrict__ __return)
{

    {


        dace::CopyND<float, 1, false, 3>::template ConstDst<3>::Copy(
        A0, __return, 1);

        dace::CopyND<float, 1, false, 3>::template ConstDst<3>::Copy(
        A1, __return + 1, 1);

        dace::CopyND<float, 1, false, 3>::template ConstDst<3>::Copy(
        A2, __return + 2, 1);

    }
}

DACE_EXPORTED void __program_recombine(recombine_t *__state, float * __restrict__ A0, float * __restrict__ A1, float * __restrict__ A2, float * __restrict__ __return)
{
    __program_recombine_internal(__state, A0, A1, A2, __return);
}

DACE_EXPORTED recombine_t *__dace_init_recombine()
{
    int __result = 0;
    recombine_t *__state = new recombine_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_recombine(recombine_t *__state)
{
    delete __state;
}


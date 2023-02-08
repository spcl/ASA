/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct pure_4_4_t {

};

void __program_pure_4_4_internal(pure_4_4_t *__state, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y)
{

    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 4; _o0 += 1) {
                {
                    float out;

                    ///////////////////
                    // Tasklet code (gemv_init)
                    out = 0;
                    ///////////////////

                    y[_o0] = out;
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < 4; __i0 += 1) {
                for (auto __i1 = 0; __i1 < 4; __i1 += 1) {
                    {
                        float __A = A[((4 * __i0) + __i1)];
                        float __x = x[__i1];
                        float __out;

                        ///////////////////
                        // Tasklet code (_GEMV_)
                        __out = ((1 * __A) * __x);
                        ///////////////////

                        dace::wcr_fixed<dace::ReductionType::Sum, float>::reduce_atomic(y + __i0, __out);
                    }
                }
            }
        }

    }
}

DACE_EXPORTED void __program_pure_4_4(pure_4_4_t *__state, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y)
{
    __program_pure_4_4_internal(__state, A, x, y);
}

DACE_EXPORTED pure_4_4_t *__dace_init_pure_4_4()
{
    int __result = 0;
    pure_4_4_t *__state = new pure_4_4_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_pure_4_4(pure_4_4_t *__state)
{
    delete __state;
}


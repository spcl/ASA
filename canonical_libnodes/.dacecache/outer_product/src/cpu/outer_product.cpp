/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct outer_product_t {

};

void __program_outer_product_internal(outer_product_t *__state, float * __restrict__ __return, float * __restrict__ u, float * __restrict__ v, int M, int N)
{

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < N; __i0 += 1) {
                for (auto __i1 = 0; __i1 < M; __i1 += 1) {
                    {
                        float __in1 = u[__i0];
                        float __in2 = v[__i1];
                        float __out;

                        ///////////////////
                        // Tasklet code (_Mult_)
                        __out = (__in1 * __in2);
                        ///////////////////

                        __return[((M * __i0) + __i1)] = __out;
                    }
                }
            }
        }

    }
}

DACE_EXPORTED void __program_outer_product(outer_product_t *__state, float * __restrict__ __return, float * __restrict__ u, float * __restrict__ v, int M, int N)
{
    __program_outer_product_internal(__state, __return, u, v, M, N);
}

DACE_EXPORTED outer_product_t *__dace_init_outer_product(int M, int N)
{
    int __result = 0;
    outer_product_t *__state = new outer_product_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_outer_product(outer_product_t *__state)
{
    delete __state;
}


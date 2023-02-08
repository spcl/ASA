/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct pure_2_2_2_t {

};

inline void MMM_sdfg_0_0_3(pure_2_2_2_t *__state, float* _a, float* _b, float* _c) {

    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 2; _o0 += 1) {
                for (auto _o1 = 0; _o1 < 2; _o1 += 1) {
                    {
                        float out;

                        ///////////////////
                        // Tasklet code (mmm_init)
                        out = 0;
                        ///////////////////

                        _c[((2 * _o0) + _o1)] = out;
                    }
                }
            }
        }

    }
    {

        {
            for (auto __i0 = 0; __i0 < 2; __i0 += 1) {
                for (auto __i1 = 0; __i1 < 2; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < 2; __i2 += 1) {
                        {
                            float __a = _a[((2 * __i0) + __i2)];
                            float __b = _b[(__i1 + (2 * __i2))];
                            float __out;

                            ///////////////////
                            // Tasklet code (mmm)
                            __out = (__a * __b);
                            ///////////////////

                            dace::wcr_fixed<dace::ReductionType::Sum, float>::reduce(_c + ((2 * __i0) + __i1), __out);
                        }
                    }
                }
            }
        }

    }
    
}

void __program_pure_2_2_2_internal(pure_2_2_2_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{

    {

        MMM_sdfg_0_0_3(__state, &A[0], &B[0], &C[0]);

    }
}

DACE_EXPORTED void __program_pure_2_2_2(pure_2_2_2_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{
    __program_pure_2_2_2_internal(__state, A, B, C);
}

DACE_EXPORTED pure_2_2_2_t *__dace_init_pure_2_2_2()
{
    int __result = 0;
    pure_2_2_2_t *__state = new pure_2_2_2_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_pure_2_2_2(pure_2_2_2_t *__state)
{
    delete __state;
}


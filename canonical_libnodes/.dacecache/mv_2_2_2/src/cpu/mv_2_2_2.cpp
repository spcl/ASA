/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct mv_2_2_2_t {

};

inline void mv_sdfg_1_1_0_3(mv_2_2_2_t *__state, float* _A, float* _x, float* _y) {

    {

        {
            for (auto i = 0; i < 2; i += 1) {
                float accum;
                {
                    float acc_out;

                    ///////////////////
                    // Tasklet code (init_accum)
                    acc_out = 0;
                    ///////////////////

                    accum = acc_out;
                }
                {
                    for (auto j = 0; j < 2; j += 1) {
                        {
                            float A_in = _A[((2 * i) + j)];
                            float x_in = _x[(2 * j)];
                            float accum_in = accum;
                            float accum_out;

                            ///////////////////
                            // Tasklet code (multiply)
                            accum_out = ((A_in * x_in) + accum_in);
                            ///////////////////

                            accum = accum_out;
                        }
                    }
                }

                dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
                &accum, _y + (2 * i), 1);
            }
        }

    }
    
}

inline void MMM_sdfg_0_0_3(mv_2_2_2_t *__state, float* _a, float* _b, float* _c) {

    {

        mv_sdfg_1_1_0_3(__state, &_a[0], &_b[0], &_c[0]);
        mv_sdfg_1_1_0_3(__state, &_a[0], &_b[1], &_c[1]);

    }
    
}

void __program_mv_2_2_2_internal(mv_2_2_2_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{

    {

        MMM_sdfg_0_0_3(__state, &A[0], &B[0], &C[0]);

    }
}

DACE_EXPORTED void __program_mv_2_2_2(mv_2_2_2_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{
    __program_mv_2_2_2_internal(__state, A, B, C);
}

DACE_EXPORTED mv_2_2_2_t *__dace_init_mv_2_2_2()
{
    int __result = 0;
    mv_2_2_2_t *__state = new mv_2_2_2_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_mv_2_2_2(mv_2_2_2_t *__state)
{
    delete __state;
}


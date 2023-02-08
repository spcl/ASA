/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct LMV_col_2_5_t {

};

void __program_LMV_col_2_5_internal(LMV_col_2_5_t *__state, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y)
{

    {

        {
            for (auto k = 0; k < 5; k += 1) {
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
                            float A_in = A[(j + (2 * k))];
                            float x_in = x[j];
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
                &accum, y + k, 1);
            }
        }

    }
}

DACE_EXPORTED void __program_LMV_col_2_5(LMV_col_2_5_t *__state, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y)
{
    __program_LMV_col_2_5_internal(__state, A, x, y);
}

DACE_EXPORTED LMV_col_2_5_t *__dace_init_LMV_col_2_5()
{
    int __result = 0;
    LMV_col_2_5_t *__state = new LMV_col_2_5_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_LMV_col_2_5(LMV_col_2_5_t *__state)
{
    delete __state;
}


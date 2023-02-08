/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct seq_3_5_t {

};

void __program_seq_3_5_internal(seq_3_5_t *__state, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y)
{

    {

        {
            for (auto i = 0; i < 3; i += 1) {
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
                    for (auto j = 0; j < 5; j += 1) {
                        {
                            float A_in = A[((5 * i) + j)];
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
                &accum, y + i, 1);
            }
        }

    }
}

DACE_EXPORTED void __program_seq_3_5(seq_3_5_t *__state, float * __restrict__ A, float * __restrict__ x, float * __restrict__ y)
{
    __program_seq_3_5_internal(__state, A, x, y);
}

DACE_EXPORTED seq_3_5_t *__dace_init_seq_3_5()
{
    int __result = 0;
    seq_3_5_t *__state = new seq_3_5_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_seq_3_5(seq_3_5_t *__state)
{
    delete __state;
}


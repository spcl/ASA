/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct gather_4_5_t {

};

void __program_gather_4_5_internal(gather_4_5_t *__state, float * __restrict__ inp, float * __restrict__ outp)
{

    {

        {
            #pragma omp parallel for
            for (auto j = 0; j < 5; j += 1) {
                for (auto i = 0; i < 4; i += 1) {
                    {
                        float _in_val = inp[((5 * i) + j)];
                        float _out_val;

                        ///////////////////
                        // Tasklet code (gather_tasklet)
                        _out_val = _in_val;
                        ///////////////////

                        outp[((5 * i) + j)] = _out_val;
                    }
                }
            }
        }

    }
}

DACE_EXPORTED void __program_gather_4_5(gather_4_5_t *__state, float * __restrict__ inp, float * __restrict__ outp)
{
    __program_gather_4_5_internal(__state, inp, outp);
}

DACE_EXPORTED gather_4_5_t *__dace_init_gather_4_5()
{
    int __result = 0;
    gather_4_5_t *__state = new gather_4_5_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_gather_4_5(gather_4_5_t *__state)
{
    delete __state;
}


/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct OP_by_col_4_5_t {

};

void __program_OP_by_col_4_5_internal(OP_by_col_4_5_t *__state, float * __restrict__ A, float * __restrict__ u, float * __restrict__ v)
{

    {

        {
            #pragma omp parallel for
            for (auto j = 0; j < 5; j += 1) {
                for (auto i = 0; i < 4; i += 1) {
                    {
                        float u_in = u[i];
                        float v_in = v[j];
                        float A_out;

                        ///////////////////
                        // Tasklet code (multiply)
                        A_out = (u_in * v_in);
                        ///////////////////

                        A[((5 * i) + j)] = A_out;
                    }
                }
            }
        }

    }
}

DACE_EXPORTED void __program_OP_by_col_4_5(OP_by_col_4_5_t *__state, float * __restrict__ A, float * __restrict__ u, float * __restrict__ v)
{
    __program_OP_by_col_4_5_internal(__state, A, u, v);
}

DACE_EXPORTED OP_by_col_4_5_t *__dace_init_OP_by_col_4_5()
{
    int __result = 0;
    OP_by_col_4_5_t *__state = new OP_by_col_4_5_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_OP_by_col_4_5(OP_by_col_4_5_t *__state)
{
    delete __state;
}


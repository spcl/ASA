/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct add_test_add_test_t {

};

void __program_add_test_add_test_internal(add_test_add_test_t *__state, float * __restrict__ A, float * __restrict__ B, int M, int N)
{

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < N; __i0 += 1) {
                for (auto __i1 = 0; __i1 < M; __i1 += 1) {
                    {
                        float __in1 = A[((M * __i0) + __i1)];
                        float __out;

                        ///////////////////
                        // Tasklet code (_Add_)
                        __out = (__in1 + dace::float32(1));
                        ///////////////////

                        B[((M * __i0) + __i1)] = __out;
                    }
                }
            }
        }

    }
}

DACE_EXPORTED void __program_add_test_add_test(add_test_add_test_t *__state, float * __restrict__ A, float * __restrict__ B, int M, int N)
{
    __program_add_test_add_test_internal(__state, A, B, M, N);
}

DACE_EXPORTED add_test_add_test_t *__dace_init_add_test_add_test(int M, int N)
{
    int __result = 0;
    add_test_add_test_t *__state = new add_test_add_test_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_add_test_add_test(add_test_add_test_t *__state)
{
    delete __state;
}


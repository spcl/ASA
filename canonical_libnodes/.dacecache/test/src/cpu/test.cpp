/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct test_t {

};

void __program_test_internal(test_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ R)
{

    {
        float *Z;
        Z = new float DACE_ALIGN(64)[32];
        float* _x;
        _x = &A[0];
        float* _x_0;
        _x_0 = &A[1];
        float* _x_1;
        _x_1 = &A[2];
        float* _x_2;
        _x_2 = &A[3];

        {
            #pragma omp parallel for
            for (auto j = 0; j < 4; j += 1) {
                for (auto i = 0; i < 2; i += 1) {
                    {
                        float x_in = _x[i];
                        float y_in = B[j];
                        float A_out;

                        ///////////////////
                        // Tasklet code (multiply)
                        A_out = (x_in * y_in);
                        ///////////////////

                        Z[((4 * i) + j)] = A_out;
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto j = 0; j < 4; j += 1) {
                for (auto i = 0; i < 2; i += 1) {
                    {
                        float x_in = _x_0[i];
                        float y_in = B[(j + 4)];
                        float A_out;

                        ///////////////////
                        // Tasklet code (multiply)
                        A_out = (x_in * y_in);
                        ///////////////////

                        Z[(((4 * i) + j) + 8)] = A_out;
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto j = 0; j < 4; j += 1) {
                for (auto i = 0; i < 2; i += 1) {
                    {
                        float x_in = _x_1[i];
                        float y_in = B[(j + 8)];
                        float A_out;

                        ///////////////////
                        // Tasklet code (multiply)
                        A_out = (x_in * y_in);
                        ///////////////////

                        Z[(((4 * i) + j) + 16)] = A_out;
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto j = 0; j < 4; j += 1) {
                for (auto i = 0; i < 2; i += 1) {
                    {
                        float x_in = _x_2[i];
                        float y_in = B[(j + 12)];
                        float A_out;

                        ///////////////////
                        // Tasklet code (multiply)
                        A_out = (x_in * y_in);
                        ///////////////////

                        Z[(((4 * i) + j) + 24)] = A_out;
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto j = 0; j < 4; j += 1) {
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
                        for (auto k = 0; k < 4; k += 1) {
                            {
                                float _accum = accum;
                                float _in = Z[(((4 * i) + j) + (8 * k))];
                                float _out;

                                ///////////////////
                                // Tasklet code (sum)
                                _out = (_in + _accum);
                                ///////////////////

                                accum = _out;
                            }
                        }
                    }

                    dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
                    &accum, R + ((4 * i) + j), 1);
                }
            }
        }
        delete[] Z;

    }
}

DACE_EXPORTED void __program_test(test_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ R)
{
    __program_test_internal(__state, A, B, R);
}

DACE_EXPORTED test_t *__dace_init_test()
{
    int __result = 0;
    test_t *__state = new test_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_test(test_t *__state)
{
    delete __state;
}


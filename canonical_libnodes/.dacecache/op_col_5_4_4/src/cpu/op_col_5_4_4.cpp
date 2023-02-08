/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct op_col_5_4_4_t {

};

void __program_op_col_5_4_4_internal(op_col_5_4_4_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, float * __restrict__ D)
{

    {
        float _partial_results[80]  DACE_ALIGN(64);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                {
                    #pragma omp parallel for
                    for (auto j = 0; j < 4; j += 1) {
                        for (auto i = 0; i < 5; i += 1) {
                            {
                                float u_in = A[(4 * i)];
                                float v_in = B[j];
                                float A_out;

                                ///////////////////
                                // Tasklet code (multiply)
                                A_out = (u_in * v_in);
                                ///////////////////

                                _partial_results[((4 * i) + j)] = A_out;
                            }
                        }
                    }
                }
            } // End omp section
            #pragma omp section
            {
                {
                    #pragma omp parallel for
                    for (auto j = 0; j < 4; j += 1) {
                        for (auto i = 0; i < 5; i += 1) {
                            {
                                float u_in = A[((4 * i) + 1)];
                                float v_in = B[(j + 4)];
                                float A_out;

                                ///////////////////
                                // Tasklet code (multiply)
                                A_out = (u_in * v_in);
                                ///////////////////

                                _partial_results[(((4 * i) + j) + 20)] = A_out;
                            }
                        }
                    }
                }
            } // End omp section
            #pragma omp section
            {
                {
                    #pragma omp parallel for
                    for (auto j = 0; j < 4; j += 1) {
                        for (auto i = 0; i < 5; i += 1) {
                            {
                                float u_in = A[((4 * i) + 2)];
                                float v_in = B[(j + 8)];
                                float A_out;

                                ///////////////////
                                // Tasklet code (multiply)
                                A_out = (u_in * v_in);
                                ///////////////////

                                _partial_results[(((4 * i) + j) + 40)] = A_out;
                            }
                        }
                    }
                }
            } // End omp section
            #pragma omp section
            {
                {
                    #pragma omp parallel for
                    for (auto j = 0; j < 4; j += 1) {
                        for (auto i = 0; i < 5; i += 1) {
                            {
                                float u_in = A[((4 * i) + 3)];
                                float v_in = B[(j + 12)];
                                float A_out;

                                ///////////////////
                                // Tasklet code (multiply)
                                A_out = (u_in * v_in);
                                ///////////////////

                                _partial_results[(((4 * i) + j) + 60)] = A_out;
                            }
                        }
                    }
                }
                {
                    #pragma omp parallel for
                    for (auto j = 0; j < 4; j += 1) {
                        for (auto i = 0; i < 5; i += 1) {
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
                                        float _in = _partial_results[(((4 * i) + j) + (20 * k))];
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
                            &accum, C + ((4 * i) + j), 1);
                        }
                    }
                }
                {
                    #pragma omp parallel for
                    for (auto _o0 = 0; _o0 < 5; _o0 += 1) {
                        for (auto _o1 = 0; _o1 < 4; _o1 += 1) {
                            {
                                float C_in = C[((4 * _o0) + _o1)];
                                float C_out;

                                ///////////////////
                                // Tasklet code (add_one)
                                C_out = (C_in + 1);
                                ///////////////////

                                D[((4 * _o0) + _o1)] = C_out;
                            }
                        }
                    }
                }
            } // End omp section
        } // End omp sections

    }
}

DACE_EXPORTED void __program_op_col_5_4_4(op_col_5_4_4_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, float * __restrict__ D)
{
    __program_op_col_5_4_4_internal(__state, A, B, C, D);
}

DACE_EXPORTED op_col_5_4_4_t *__dace_init_op_col_5_4_4()
{
    int __result = 0;
    op_col_5_4_4_t *__state = new op_col_5_4_4_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_op_col_5_4_4(op_col_5_4_4_t *__state)
{
    delete __state;
}


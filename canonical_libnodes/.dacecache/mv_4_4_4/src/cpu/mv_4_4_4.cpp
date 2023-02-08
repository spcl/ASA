/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct mv_4_4_4_t {

};

void __program_mv_4_4_4_internal(mv_4_4_4_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, float * __restrict__ D)
{

    {
        float _tmp[16]  DACE_ALIGN(64);
        dace::Stream<float> stream_C(1);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                {
                    for (auto i = 0; i < 4; i += 1) {
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
                            for (auto j = 0; j < 4; j += 1) {
                                {
                                    float A_in = A[((4 * i) + j)];
                                    float x_in = B[(4 * j)];
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
                        &accum, _tmp + (4 * i), 1);
                    }
                }
                {
                    for (auto i = 0; i < 4; i += 1) {
                        float mv_sdfg_1_accum;
                        {
                            float acc_out;

                            ///////////////////
                            // Tasklet code (init_accum)
                            acc_out = 0;
                            ///////////////////

                            mv_sdfg_1_accum = acc_out;
                        }
                        {
                            for (auto j = 0; j < 4; j += 1) {
                                {
                                    float A_in = A[((4 * i) + j)];
                                    float x_in = B[((4 * j) + 1)];
                                    float accum_in = mv_sdfg_1_accum;
                                    float accum_out;

                                    ///////////////////
                                    // Tasklet code (multiply)
                                    accum_out = ((A_in * x_in) + accum_in);
                                    ///////////////////

                                    mv_sdfg_1_accum = accum_out;
                                }
                            }
                        }

                        dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
                        &mv_sdfg_1_accum, _tmp + ((4 * i) + 1), 1);
                    }
                }
                {
                    for (auto i = 0; i < 4; i += 1) {
                        float mv_sdfg_2_accum;
                        {
                            float acc_out;

                            ///////////////////
                            // Tasklet code (init_accum)
                            acc_out = 0;
                            ///////////////////

                            mv_sdfg_2_accum = acc_out;
                        }
                        {
                            for (auto j = 0; j < 4; j += 1) {
                                {
                                    float A_in = A[((4 * i) + j)];
                                    float x_in = B[((4 * j) + 2)];
                                    float accum_in = mv_sdfg_2_accum;
                                    float accum_out;

                                    ///////////////////
                                    // Tasklet code (multiply)
                                    accum_out = ((A_in * x_in) + accum_in);
                                    ///////////////////

                                    mv_sdfg_2_accum = accum_out;
                                }
                            }
                        }

                        dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
                        &mv_sdfg_2_accum, _tmp + ((4 * i) + 2), 1);
                    }
                }
                {
                    for (auto i = 0; i < 4; i += 1) {
                        float mv_sdfg_3_accum;
                        {
                            float acc_out;

                            ///////////////////
                            // Tasklet code (init_accum)
                            acc_out = 0;
                            ///////////////////

                            mv_sdfg_3_accum = acc_out;
                        }
                        {
                            for (auto j = 0; j < 4; j += 1) {
                                {
                                    float A_in = A[((4 * i) + j)];
                                    float x_in = B[((4 * j) + 3)];
                                    float accum_in = mv_sdfg_3_accum;
                                    float accum_out;

                                    ///////////////////
                                    // Tasklet code (multiply)
                                    accum_out = ((A_in * x_in) + accum_in);
                                    ///////////////////

                                    mv_sdfg_3_accum = accum_out;
                                }
                            }
                        }

                        dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
                        &mv_sdfg_3_accum, _tmp + ((4 * i) + 3), 1);
                    }
                }
                {
                    #pragma omp parallel for
                    for (auto i = 0; i < 4; i += 1) {
                        for (auto j = 0; j < 4; j += 1) {
                            {
                                float _in = _tmp[((4 * i) + j)];

                                ///////////////////
                                // Tasklet code (copy_out)
                                stream_C.push(_in);
                                ///////////////////

                            }
                        }
                    }
                }
            } // End omp section
            #pragma omp section
            {
                {
                    #pragma omp parallel for
                    for (auto _o0 = 0; _o0 < 4; _o0 += 1) {
                        for (auto _o1 = 0; _o1 < 4; _o1 += 1) {
                            {
                                float C_in = (stream_C).pop();
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

DACE_EXPORTED void __program_mv_4_4_4(mv_4_4_4_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, float * __restrict__ D)
{
    __program_mv_4_4_4_internal(__state, A, B, C, D);
}

DACE_EXPORTED mv_4_4_4_t *__dace_init_mv_4_4_4()
{
    int __result = 0;
    mv_4_4_4_t *__state = new mv_4_4_4_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_mv_4_4_4(mv_4_4_4_t *__state)
{
    delete __state;
}


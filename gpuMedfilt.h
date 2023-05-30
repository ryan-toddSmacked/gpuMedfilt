#ifndef __GPU_MEDFILT_CU_DECL_H
#define __GPU_MEDFILT_CU_DECL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>


typedef enum
{
    int8 = 0,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64
} dtype_t;

/**
 * @brief Calculate the median filter of the input ptr, placing the result in dst
 * @param src Input pointer, ROW-MAJOR
 * @param dst Output pointer, ROW-MAJOR
 * @param dtype Primitive data type of input, and output.
 * @param M Number of rows in input and output matrices.
 * @param N Number of columns in input and output matrices.
 * @param win Square side length of window, can be one of the following, 3 | 5 | 7 | 9 | 11.
 * @param threads Value used to determine the number of threads to use in GPU kernel, the actual number of threads will be squared as we use the x and y direction for the threads argument in the kernel.
 */
int medfilt2(const void* src, void* dst, dtype_t dtype, int M, int N, int win, int threads);

#ifdef __cplusplus
}
#endif

#endif


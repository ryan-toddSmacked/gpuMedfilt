#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

#include "gpuMedfilt.h"

#define min(A,B) ((A)<(B) ? (A) : (B))
#define max(A,B) ((A)<(B) ? (B) : (A))


static const size_t dtype_size[10] = {
    sizeof(int8_t),
    sizeof(int16_t),
    sizeof(int32_t),
    sizeof(int64_t),
    sizeof(uint8_t),
    sizeof(uint16_t),
    sizeof(uint32_t),
    sizeof(uint64_t),
    sizeof(float),
    sizeof(double)
};

inline void utc_time(FILE* stream)
{
    time_t raw_time;
    struct tm *time_info;
    char utcTime[21] = {0};
    // Get the current time in UTC
    time(&raw_time);
    time_info = gmtime(&raw_time);

    // Format the time string according to ISO 8601
    strftime(utcTime, sizeof(char) * 21, "%Y-%m-%dT%H:%M:%SZ", time_info);
    fprintf(stream, "[%s]", utcTime);
}

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
inline int gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        utc_time(stderr);
        fprintf(stderr," GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        return 1;
    }
    return 0;
}


__global__ void medianFilter_kernel_i8_3x3(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i8_5x5(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i8_7x7(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i8_9x9(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i8_11x11(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i16_3x3(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i16_5x5(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i16_7x7(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i16_9x9(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i16_11x11(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i32_3x3(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i32_5x5(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i32_7x7(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i32_9x9(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i32_11x11(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i64_3x3(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i64_5x5(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i64_7x7(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i64_9x9(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_i64_11x11(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u8_3x3(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u8_5x5(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u8_7x7(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u8_9x9(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u8_11x11(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u16_3x3(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u16_5x5(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u16_7x7(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u16_9x9(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u16_11x11(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u32_3x3(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u32_5x5(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u32_7x7(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u32_9x9(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u32_11x11(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u64_3x3(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u64_5x5(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u64_7x7(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u64_9x9(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_u64_11x11(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f32_3x3(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f32_5x5(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f32_7x7(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f32_9x9(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f32_11x11(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f64_3x3(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f64_5x5(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f64_7x7(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f64_9x9(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride);
__global__ void medianFilter_kernel_f64_11x11(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride);


int medfilt2(const void* src, void* dst, dtype_t dtype, int M, int N, int win, int threads)
{
    if (!(win == 3 || win == 5 || win == 7 || win == 9 || win == 11))
    {
        utc_time(stderr);
        fprintf(stderr, " Window size must be one of the following {3,5,7,9,11}\n");
        return 0;
    }

    const size_t dtypeSize = dtype_size[dtype];

    void* d_src, *d_dst;
    size_t d_src_pitch, d_dst_pitch;
    if (gpuErrchk(cudaMallocPitch((void**)&d_src, &d_src_pitch, dtypeSize * N, M)))
    {
        return 0;
    }
    if (gpuErrchk(cudaMallocPitch((void**)&d_dst, &d_dst_pitch, dtypeSize * N, M)))
    {
        cudaFree(d_src);
        return 0;
    }

    if (gpuErrchk(cudaMemcpy2D((void*)d_src, d_src_pitch, (const void*)src, dtypeSize * N, dtypeSize * N, M, cudaMemcpyHostToDevice)))
    {
        cudaFree(d_src);
        cudaFree(d_dst);
        return 0;
    }

    dim3 threadsPerBlock(threads,threads);
    dim3 grid((unsigned int)ceil(N / (double)threadsPerBlock.x), (unsigned int)ceil(M / (double)threadsPerBlock.y));

    // Call correct kernel
    switch (dtype)
    {
        case int8:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_i8_3x3<<<grid, threadsPerBlock>>>((const int8_t*)d_src, (int8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_i8_5x5<<<grid, threadsPerBlock>>>((const int8_t*)d_src, (int8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_i8_7x7<<<grid, threadsPerBlock>>>((const int8_t*)d_src, (int8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_i8_9x9<<<grid, threadsPerBlock>>>((const int8_t*)d_src, (int8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_i8_11x11<<<grid, threadsPerBlock>>>((const int8_t*)d_src, (int8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
    
        case int16:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_i16_3x3<<<grid, threadsPerBlock>>>((const int16_t*)d_src, (int16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_i16_5x5<<<grid, threadsPerBlock>>>((const int16_t*)d_src, (int16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_i16_7x7<<<grid, threadsPerBlock>>>((const int16_t*)d_src, (int16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_i16_9x9<<<grid, threadsPerBlock>>>((const int16_t*)d_src, (int16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_i16_11x11<<<grid, threadsPerBlock>>>((const int16_t*)d_src, (int16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
        
        case int32:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_i32_3x3<<<grid, threadsPerBlock>>>((const int32_t*)d_src, (int32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_i32_5x5<<<grid, threadsPerBlock>>>((const int32_t*)d_src, (int32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_i32_7x7<<<grid, threadsPerBlock>>>((const int32_t*)d_src, (int32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_i32_9x9<<<grid, threadsPerBlock>>>((const int32_t*)d_src, (int32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_i32_11x11<<<grid, threadsPerBlock>>>((const int32_t*)d_src, (int32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;

        case int64:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_i64_3x3<<<grid, threadsPerBlock>>>((const int64_t*)d_src, (int64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_i64_5x5<<<grid, threadsPerBlock>>>((const int64_t*)d_src, (int64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_i64_7x7<<<grid, threadsPerBlock>>>((const int64_t*)d_src, (int64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_i64_9x9<<<grid, threadsPerBlock>>>((const int64_t*)d_src, (int64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_i64_11x11<<<grid, threadsPerBlock>>>((const int64_t*)d_src, (int64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
        
        case uint8:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_u8_3x3<<<grid, threadsPerBlock>>>((const uint8_t*)d_src, (uint8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_u8_5x5<<<grid, threadsPerBlock>>>((const uint8_t*)d_src, (uint8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_u8_7x7<<<grid, threadsPerBlock>>>((const uint8_t*)d_src, (uint8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_u8_9x9<<<grid, threadsPerBlock>>>((const uint8_t*)d_src, (uint8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_u8_11x11<<<grid, threadsPerBlock>>>((const uint8_t*)d_src, (uint8_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
        
        case uint16:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_u16_3x3<<<grid, threadsPerBlock>>>((const uint16_t*)d_src, (uint16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_u16_5x5<<<grid, threadsPerBlock>>>((const uint16_t*)d_src, (uint16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_u16_7x7<<<grid, threadsPerBlock>>>((const uint16_t*)d_src, (uint16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_u16_9x9<<<grid, threadsPerBlock>>>((const uint16_t*)d_src, (uint16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_u16_11x11<<<grid, threadsPerBlock>>>((const uint16_t*)d_src, (uint16_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
        
        case uint32:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_u32_3x3<<<grid, threadsPerBlock>>>((const uint32_t*)d_src, (uint32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_u32_5x5<<<grid, threadsPerBlock>>>((const uint32_t*)d_src, (uint32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_u32_7x7<<<grid, threadsPerBlock>>>((const uint32_t*)d_src, (uint32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_u32_9x9<<<grid, threadsPerBlock>>>((const uint32_t*)d_src, (uint32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_u32_11x11<<<grid, threadsPerBlock>>>((const uint32_t*)d_src, (uint32_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
        
        case uint64:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_u64_3x3<<<grid, threadsPerBlock>>>((const uint64_t*)d_src, (uint64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_u64_5x5<<<grid, threadsPerBlock>>>((const uint64_t*)d_src, (uint64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_u64_7x7<<<grid, threadsPerBlock>>>((const uint64_t*)d_src, (uint64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_u64_9x9<<<grid, threadsPerBlock>>>((const uint64_t*)d_src, (uint64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_u64_11x11<<<grid, threadsPerBlock>>>((const uint64_t*)d_src, (uint64_t*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;

        case float32:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_f32_3x3<<<grid, threadsPerBlock>>>((const float*)d_src, (float*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_f32_5x5<<<grid, threadsPerBlock>>>((const float*)d_src, (float*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_f32_7x7<<<grid, threadsPerBlock>>>((const float*)d_src, (float*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_f32_9x9<<<grid, threadsPerBlock>>>((const float*)d_src, (float*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_f32_11x11<<<grid, threadsPerBlock>>>((const float*)d_src, (float*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
        
        case float64:
            switch (win)
            {
                case 3:
                    medianFilter_kernel_f64_3x3<<<grid, threadsPerBlock>>>((const double*)d_src, (double*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 5:
                    medianFilter_kernel_f64_5x5<<<grid, threadsPerBlock>>>((const double*)d_src, (double*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 7:
                    medianFilter_kernel_f64_7x7<<<grid, threadsPerBlock>>>((const double*)d_src, (double*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 9:
                    medianFilter_kernel_f64_9x9<<<grid, threadsPerBlock>>>((const double*)d_src, (double*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
                case 11:
                    medianFilter_kernel_f64_11x11<<<grid, threadsPerBlock>>>((const double*)d_src, (double*)d_dst, M, N, d_src_pitch, d_dst_pitch);
                    break;
            };
            break;
        
        default:
            utc_time(stderr);
            fprintf(stderr, " Unrecognized primitive data type in medfilt2\n");
            cudaFree(d_src);
            cudaFree(d_dst);
            return 0;
    };

    if (gpuErrchk(cudaMemcpy2D((void*)dst, dtypeSize * N, (const void*)d_dst, d_dst_pitch, dtypeSize * N, M, cudaMemcpyDeviceToHost)))
    {
        cudaFree(d_src);
        cudaFree(d_dst);
        return 0;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return 1;
}



__global__ void medianFilter_kernel_i8_3x3(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int8_t);
    const int dst_cols = dst_stride / sizeof(int8_t);
    int8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i8_5x5(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int8_t);
    const int dst_cols = dst_stride / sizeof(int8_t);
    int8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i8_7x7(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int8_t);
    const int dst_cols = dst_stride / sizeof(int8_t);
    int8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i8_9x9(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int8_t);
    const int dst_cols = dst_stride / sizeof(int8_t);
    int8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i8_11x11(const int8_t* pSrc, int8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int8_t);
    const int dst_cols = dst_stride / sizeof(int8_t);
    int8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i16_3x3(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int16_t);
    const int dst_cols = dst_stride / sizeof(int16_t);
    int16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i16_5x5(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int16_t);
    const int dst_cols = dst_stride / sizeof(int16_t);
    int16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i16_7x7(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int16_t);
    const int dst_cols = dst_stride / sizeof(int16_t);
    int16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i16_9x9(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int16_t);
    const int dst_cols = dst_stride / sizeof(int16_t);
    int16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i16_11x11(const int16_t* pSrc, int16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int16_t);
    const int dst_cols = dst_stride / sizeof(int16_t);
    int16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i32_3x3(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int32_t);
    const int dst_cols = dst_stride / sizeof(int32_t);
    int32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i32_5x5(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int32_t);
    const int dst_cols = dst_stride / sizeof(int32_t);
    int32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i32_7x7(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int32_t);
    const int dst_cols = dst_stride / sizeof(int32_t);
    int32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i32_9x9(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int32_t);
    const int dst_cols = dst_stride / sizeof(int32_t);
    int32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i32_11x11(const int32_t* pSrc, int32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int32_t);
    const int dst_cols = dst_stride / sizeof(int32_t);
    int32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i64_3x3(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int64_t);
    const int dst_cols = dst_stride / sizeof(int64_t);
    int64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i64_5x5(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int64_t);
    const int dst_cols = dst_stride / sizeof(int64_t);
    int64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i64_7x7(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int64_t);
    const int dst_cols = dst_stride / sizeof(int64_t);
    int64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i64_9x9(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int64_t);
    const int dst_cols = dst_stride / sizeof(int64_t);
    int64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_i64_11x11(const int64_t* pSrc, int64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(int64_t);
    const int dst_cols = dst_stride / sizeof(int64_t);
    int64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                int64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u8_3x3(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint8_t);
    const int dst_cols = dst_stride / sizeof(uint8_t);
    uint8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u8_5x5(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint8_t);
    const int dst_cols = dst_stride / sizeof(uint8_t);
    uint8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u8_7x7(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint8_t);
    const int dst_cols = dst_stride / sizeof(uint8_t);
    uint8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u8_9x9(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint8_t);
    const int dst_cols = dst_stride / sizeof(uint8_t);
    uint8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u8_11x11(const uint8_t* pSrc, uint8_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint8_t);
    const int dst_cols = dst_stride / sizeof(uint8_t);
    uint8_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint8_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u16_3x3(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint16_t);
    const int dst_cols = dst_stride / sizeof(uint16_t);
    uint16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u16_5x5(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint16_t);
    const int dst_cols = dst_stride / sizeof(uint16_t);
    uint16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u16_7x7(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint16_t);
    const int dst_cols = dst_stride / sizeof(uint16_t);
    uint16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u16_9x9(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint16_t);
    const int dst_cols = dst_stride / sizeof(uint16_t);
    uint16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u16_11x11(const uint16_t* pSrc, uint16_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint16_t);
    const int dst_cols = dst_stride / sizeof(uint16_t);
    uint16_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint16_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u32_3x3(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint32_t);
    const int dst_cols = dst_stride / sizeof(uint32_t);
    uint32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u32_5x5(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint32_t);
    const int dst_cols = dst_stride / sizeof(uint32_t);
    uint32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u32_7x7(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint32_t);
    const int dst_cols = dst_stride / sizeof(uint32_t);
    uint32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u32_9x9(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint32_t);
    const int dst_cols = dst_stride / sizeof(uint32_t);
    uint32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u32_11x11(const uint32_t* pSrc, uint32_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint32_t);
    const int dst_cols = dst_stride / sizeof(uint32_t);
    uint32_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint32_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u64_3x3(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint64_t);
    const int dst_cols = dst_stride / sizeof(uint64_t);
    uint64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u64_5x5(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint64_t);
    const int dst_cols = dst_stride / sizeof(uint64_t);
    uint64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u64_7x7(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint64_t);
    const int dst_cols = dst_stride / sizeof(uint64_t);
    uint64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u64_9x9(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint64_t);
    const int dst_cols = dst_stride / sizeof(uint64_t);
    uint64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_u64_11x11(const uint64_t* pSrc, uint64_t* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(uint64_t);
    const int dst_cols = dst_stride / sizeof(uint64_t);
    uint64_t pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                uint64_t tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f32_3x3(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(float);
    const int dst_cols = dst_stride / sizeof(float);
    float pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                float tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f32_5x5(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(float);
    const int dst_cols = dst_stride / sizeof(float);
    float pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                float tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f32_7x7(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(float);
    const int dst_cols = dst_stride / sizeof(float);
    float pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                float tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f32_9x9(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(float);
    const int dst_cols = dst_stride / sizeof(float);
    float pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                float tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f32_11x11(const float* pSrc, float* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(float);
    const int dst_cols = dst_stride / sizeof(float);
    float pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                float tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f64_3x3(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 3;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(double);
    const int dst_cols = dst_stride / sizeof(double);
    double pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                double tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f64_5x5(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 5;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(double);
    const int dst_cols = dst_stride / sizeof(double);
    double pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                double tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f64_7x7(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 7;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(double);
    const int dst_cols = dst_stride / sizeof(double);
    double pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                double tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f64_9x9(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 9;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(double);
    const int dst_cols = dst_stride / sizeof(double);
    double pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                double tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}

__global__ void medianFilter_kernel_f64_11x11(const double* pSrc, double* pDst, int rows, int cols, int src_stride, int dst_stride){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col >= cols) || (row >= rows))
        return;
    const int windowSize = 11;
    const int halfWindow = windowSize / 2;
    const int windowSizeTotal = windowSize * windowSize;
    const int src_cols = src_stride / sizeof(double);
    const int dst_cols = dst_stride / sizeof(double);
    double pixelValues[windowSizeTotal] = {0};
    int idx = 0;
    int i0 = min(halfWindow, row);
    int j0 = min(halfWindow, col);
    int in = min(rows - row - 1, halfWindow);
    int jn = min(cols - col - 1, halfWindow);
    for (int i = -i0; i <= in; i++)
        for (int j = -j0; j <= jn; j++)
            pixelValues[idx++] = pSrc[(row + i) * src_cols + (col + j)];
    for (int i = 0; i <= windowSizeTotal / 2; i++)
    {
        for (int j = 0; j < windowSizeTotal - i - 1; j++)
        {
            if (pixelValues[j] > pixelValues[j+1])
            {
                double tmp = pixelValues[j];
                pixelValues[j] = pixelValues[j+1];
                pixelValues[j+1] = tmp;
            }
        }
    }
    pDst[row * dst_cols + col] = pixelValues[windowSizeTotal / 2];
}


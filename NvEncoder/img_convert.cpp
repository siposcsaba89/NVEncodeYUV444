#include "img_convert.h"

//__global__ void kernel_convertBayer8ToNV12Y(const uint8_t * src, uint8_t * dst, 
//                                           int32_t width, int32_t height, int32_t pitch)
//{
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//
//    int32_t w_2 = width / 2;
//    int32_t h_2 = height / 2;
//
//    if (x >= w_2 || y >= h_2)
//        return;
//    dst[x +       pitch * y        ] = src[2 * x + 0 + (2 * y + 0)*pitch];
//    dst[x + w_2 + pitch * y        ] = src[2 * x + 1 + (2 * y + 0)*pitch];
//    dst[x +       pitch * (y + h_2)] = src[2 * x + 0 + (2 * y + 1)*pitch];
//    dst[x + w_2 + pitch * (y + h_2)] = src[2 * x + 1 + (2 * y + 1)*pitch];
//}
//
//__global__ void kernel_convertBayer8ToNV12Y_setUV(uint8_t * dst,
//                                            int32_t width, int32_t height, int32_t pitch, char u)
//{
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//
//    int32_t w_2 = width / 2;
//    int32_t h_2 = height / 2;
//
//    if (x >= w_2 || y >= h_2)
//        return;
//    dst[2*x + 0 + pitch * (y + height)] = u;
//    dst[2*x + 1 + pitch * (y + height)] = u;
//}
//
//__global__ void kernel_convertNV12YtoBayer8(const uint8_t * src, uint8_t * dst,
//                                            int32_t width, int32_t height, int32_t pitch)
//{
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//
//    int32_t w_2 = width / 2;
//    int32_t h_2 = height / 2;
//
//    if (x >= w_2 || y >= h_2)
//        return;
//    dst[2 * x + 0 + (2 * y + 0)*pitch] = src[x + pitch * y];
//    dst[2 * x + 1 + (2 * y + 0)*pitch] = src[x + w_2 + pitch * y];
//    dst[2 * x + 0 + (2 * y + 1)*pitch] = src[x + pitch * (y + h_2)];
//    dst[2 * x + 1 + (2 * y + 1)*pitch] = src[x + w_2 + pitch * (y + h_2)];
//} 

/// v is not implemented yet
void nvencc::convertBayer8ToNV12_Y(const CUdeviceptr src, CUdeviceptr dst, int32_t width, int32_t height, int32_t pitch,
                                   char u, char v)
{
}


void nvencc::convertNV12_Y_toBayer8(const CUdeviceptr src, CUdeviceptr dst, int32_t width, int32_t height, int32_t pitch)
{
}

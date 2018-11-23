#include "img_convert.h"

__global__ void kernel_convertBayer8ToNV12Y(const uint8_t * src, int32_t src_pitch, uint8_t * dst,
                                           int32_t width, int32_t height, int32_t dst_pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int32_t w_2 = width / 2;
    int32_t h_2 = height / 2;

    if (x >= w_2 || y >= h_2)
        return;
    dst[x +       dst_pitch * y        ] = src[2 * x + 0 + (2 * y + 0)*src_pitch];
    dst[x + w_2 + dst_pitch * y        ] = src[2 * x + 1 + (2 * y + 0)*src_pitch];
    dst[x +       dst_pitch * (y + h_2)] = src[2 * x + 0 + (2 * y + 1)*src_pitch];
    dst[x + w_2 + dst_pitch * (y + h_2)] = src[2 * x + 1 + (2 * y + 1)*src_pitch];
}

__global__ void kernel_convertBayer8ToNV12Y_setUV(uint8_t * dst,
                                            int32_t width, int32_t height, int32_t dst_pitch, char u)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int32_t w_2 = width / 2;
    int32_t h_2 = height / 2;

    if (x >= w_2 || y >= h_2)
        return;
    dst[2*x + 0 + dst_pitch * (y + height)] = u;
    dst[2*x + 1 + dst_pitch * (y + height)] = u;
}

__global__ void kernel_convertNV12YtoBayer8(const uint8_t * src, int32_t src_pitch, uint8_t * dst,
                                            int32_t width, int32_t height, int32_t dst_pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int32_t w_2 = width / 2;
    int32_t h_2 = height / 2;

    if (x >= w_2 || y >= h_2)
        return;
    dst[2 * x + 0 + (2 * y + 0)*dst_pitch] = src[x + src_pitch * y];
    dst[2 * x + 1 + (2 * y + 0)*dst_pitch] = src[x + w_2 + src_pitch * y];
    dst[2 * x + 0 + (2 * y + 1)*dst_pitch] = src[x + src_pitch * (y + h_2)];
    dst[2 * x + 1 + (2 * y + 1)*dst_pitch] = src[x + w_2 + src_pitch * (y + h_2)];
} 

__global__ void kernel_convertNV12YUV420toRGB(const uint8_t * src, uint32_t src_pitch, uint8_t * dst,
    uint32_t width, uint32_t height, uint32_t dst_pitch, uint32_t chc, uint8_t alpha)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= width || y >= height)
        return;

    int16_t C = int16_t(src[x + src_pitch * y]) - 16; //y
    int16_t D = int16_t(src[2*(x / 2) + 0 + src_pitch * (height + y / 2)]) - 128; //u
    int16_t E = int16_t(src[2*(x / 2) + 1 + src_pitch * (height + y / 2)]) - 128; //v
    
    // rgb
    dst[(x + 0) * chc + y * dst_pitch] = (uint8_t)min(max(round(1.164383f * C + 1.596027f * E), 0.0f), 255.0f);
    dst[(x + 1) * chc + y * dst_pitch] = (uint8_t)min(max(round(1.164383f * C - (0.391762f * D) - (0.812968f * E)), 0.0f), 255.0f);
    dst[(x + 2) * chc + y * dst_pitch] = (uint8_t)min(max(round(1.164383f * C + 2.017232f * D), 0.0f), 255.0f);
    if (chc > 3)
        dst[(x + 3) * chc + y * dst_pitch] = alpha;
}


__global__ void kernel_convertNV12YUV420toRGB_surf(const uint8_t * src, uint32_t src_pitch,
    cudaSurfaceObject_t dst,
    uint32_t width, uint32_t height, uint8_t alpha)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int16_t C = int16_t(src[x + src_pitch * y]) - 16; //y
    int16_t D = int16_t(src[2*(x / 2) + 0 + src_pitch * (height + y / 2)]) - 128; //u
    int16_t E = int16_t(src[2*(x / 2) + 1 + src_pitch * (height + y / 2)]) - 128; //v

    // rgb
    uchar4 data;
    data.x = (uint8_t)min(max(round(1.164383f * C + 1.596027f * E), 0.0f), 255.0f);
    data.y = (uint8_t)min(max(round(1.164383f * C - (0.391762f * D) - (0.812968f * E)), 0.0f), 255.0f);
    data.z = (uint8_t)min(max(round(1.164383f * C + 2.017232f * D), 0.0f), 255.0f);
    data.w = alpha;
    surf2Dwrite(data, dst, x * 4, y);
}


/// v is not implemented yet
void nvencc::convertBayer8ToNV12_Y(const CUdeviceptr src, int32_t src_pitch,
    CUdeviceptr dst, int32_t width, int32_t height, int32_t dst_pitch,
                                   char u, char v)
{
    dim3 loc_dim(16,16);
    dim3 glob_dim((width/2 + loc_dim.x - 1) / loc_dim.x , (height/2 + loc_dim.y - 1) / loc_dim.y);
    kernel_convertBayer8ToNV12Y << <glob_dim, loc_dim >> > ((const uint8_t*)src, src_pitch,
        (uint8_t*)dst, width, height, dst_pitch);
    kernel_convertBayer8ToNV12Y_setUV << <glob_dim, loc_dim >> > ((uint8_t*)dst, width, height, dst_pitch, u);
}


void nvencc::convertNV12_Y_toBayer8(const CUdeviceptr src,
    int32_t src_pitch,
    CUdeviceptr dst, int32_t width, int32_t height, int32_t dst_pitch)
{
    dim3 loc_dim(16, 16);
    dim3 glob_dim((width / 2 + loc_dim.x - 1) / loc_dim.x, (height / 2 + loc_dim.y - 1) / loc_dim.y);
    kernel_convertNV12YtoBayer8 << <glob_dim, loc_dim >> > ((const uint8_t*)src, src_pitch,
        (uint8_t*)dst, width, height, dst_pitch);
}

void nvencc::convertYUV420_toRGB(const CUdeviceptr src, uint32_t src_pitch, 
    CUdeviceptr dst, uint32_t width, uint32_t height, uint32_t dst_pitch)
{
    dim3 loc_dim(16, 16);
    dim3 glob_dim((width + loc_dim.x - 1) / loc_dim.x, (height + loc_dim.y - 1) / loc_dim.y);
    uint8_t alpha = 255;
    uint32_t chc = 3;
    dst_pitch *= chc;
    kernel_convertNV12YUV420toRGB << <glob_dim, loc_dim>> > ((const uint8_t*)src, src_pitch, 
        (uint8_t*)dst, width, height, dst_pitch, chc, alpha);
}

void nvencc::convertYUV420_toRGB(const CUdeviceptr src, uint32_t src_pitch,
    CUsurfObject dst, uint32_t width, uint32_t height)
{
    dim3 loc_dim(16, 16);
    dim3 glob_dim((width + loc_dim.x - 1) / loc_dim.x, (height + loc_dim.y - 1) / loc_dim.y);
    uint8_t alpha = 255;
    //always RGBA channel
    kernel_convertNV12YUV420toRGB_surf << <glob_dim, loc_dim >> > ((const uint8_t*)src, src_pitch,
        dst, width, height, alpha);
}

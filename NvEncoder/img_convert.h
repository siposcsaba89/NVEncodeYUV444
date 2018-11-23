#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "NvEncoder/NvEncoder.h"

namespace nvencc
{
void convertBayer8ToNV12_Y(const CUdeviceptr src,
    int32_t src_pitch,
    CUdeviceptr dst, int32_t width, int32_t height, 
    int32_t dst_pitch, char u = 127, char v = 127);
void convertNV12_Y_toBayer8(const CUdeviceptr src, 
    int32_t src_pitch,
    CUdeviceptr dst, int32_t width, int32_t height, int32_t dst_pitch);
void convertYUV420_toRGB(const CUdeviceptr src, uint32_t src_pitch,
    CUdeviceptr dst, uint32_t width, uint32_t height, uint32_t dst_pitch);
void convertYUV420_toRGB(const CUdeviceptr src, uint32_t src_pitch,
    CUsurfObject dst, uint32_t width, uint32_t height);

}
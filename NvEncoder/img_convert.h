#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "NvEncoder/NvEncoder.h"

namespace nvencc
{
void convertBayer8ToNV12_Y(const CUdeviceptr src, CUdeviceptr dst, int32_t width, int32_t height, int32_t pitch, char u = 127, char v = 127);
void convertNV12_Y_toBayer8(const CUdeviceptr src, CUdeviceptr dst, int32_t width, int32_t height, int32_t pitch);
}
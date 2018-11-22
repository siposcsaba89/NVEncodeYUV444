////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>
#if defined(NV_WINDOWS)
    #include <d3d9.h>
    #include <d3d10_1.h>
    #include <d3d11.h>
#pragma warning(disable : 4996)
#endif

#include "NvEncoder/NvHWEncoder.h"

#define MAX_ENCODE_QUEUE 32
#define FRAME_QUEUE 240
#define NUM_OF_MVHINTS_PER_BLOCK8x8   4
#define NUM_OF_MVHINTS_PER_BLOCK8x16  2
#define NUM_OF_MVHINTS_PER_BLOCK16x8  2
#define NUM_OF_MVHINTS_PER_BLOCK16x16 1

enum
{
    PARTITION_TYPE_16x16,
    PARTITION_TYPE_8x8,
    PARTITION_TYPE_16x8,
    PARTITION_TYPE_8x16
};
#define SET_VER(configStruct, type) {configStruct.version = type##_VER;}

template<class T>
class CNvQueue {
    T** m_pBuffer;
    unsigned int m_uSize;
    unsigned int m_uPendingCount;
    unsigned int m_uAvailableIdx;
    unsigned int m_uPendingndex;
public:
    CNvQueue(): m_pBuffer(NULL), m_uSize(0), m_uPendingCount(0), m_uAvailableIdx(0),
                m_uPendingndex(0)
    {
    }

    ~CNvQueue()
    {
        delete[] m_pBuffer;
    }

    bool Initialize(T *pItems, unsigned int uSize)
    {
        m_uSize = uSize;
        m_uPendingCount = 0;
        m_uAvailableIdx = 0;
        m_uPendingndex = 0;
        m_pBuffer = new T *[m_uSize];
        for (unsigned int i = 0; i < m_uSize; i++)
        {
            m_pBuffer[i] = &pItems[i];
        }
        return true;
    }


    T * GetAvailable()
    {
        T *pItem = NULL;
        if (m_uPendingCount == m_uSize)
        {
            return NULL;
        }
        pItem = m_pBuffer[m_uAvailableIdx];
        m_uAvailableIdx = (m_uAvailableIdx+1)%m_uSize;
        m_uPendingCount += 1;
        return pItem;
    }

    T* GetPending()
    {
        if (m_uPendingCount == 0) 
        {
            return NULL;
        }

        T *pItem = m_pBuffer[m_uPendingndex];
        m_uPendingndex = (m_uPendingndex+1)%m_uSize;
        m_uPendingCount -= 1;
        return pItem;
    }
};

typedef enum
{
    NV_ENC_BAYER8 = 0,
    NV_ENC_NV12 = 1
} NvEncodeImageType;

typedef struct _EncodeFrameConfig
{
    CUdeviceptr input_ptr;
    NvEncodeImageType img_type;
    uint32_t stride[3];
    uint32_t width;
    uint32_t height;
    int8_t *qpDeltaMapArray;
    uint32_t qpDeltaMapArraySize;
    NVENC_EXTERNAL_ME_HINT *meExternalHints;
    NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE meHintCountsPerBlock[1];
}EncodeFrameConfig;

typedef enum 
{
    NV_ENC_DX9 = 0,
    NV_ENC_DX11 = 1,
    NV_ENC_CUDA = 2,
    NV_ENC_DX10 = 3,
} NvEncodeDeviceType;



class CNvEncoder
{
public:
    CNvEncoder();
    virtual ~CNvEncoder();

    void init(int w, int h, const std::vector<std::string> & output_files,
        const std::string & externalHintInputFile);

    bool encodeFrame(CUdeviceptr input_img, NvEncodeImageType img_type, int cam_idx, int frame_idx);
    EncodeConfig encodeConfig;

protected:
    CNvHWEncoder * m_pNvHWEncoder;
    uint32_t m_uEncodeBufferCount;
    uint32_t m_uPicStruct;
    void* m_pDevice;
#if defined(NV_WINDOWS)
    //IDirect3D9  *m_pD3D;
#endif
    CUcontext m_cuContext;
    EncodeConfig m_stEncoderInput;
    std::vector<std::vector<EncodeBuffer>> m_stEncodeBuffer;// [MAX_ENCODE_QUEUE];
    std::vector<std::vector<MotionEstimationBuffer>> m_stMVBuffer;// [MAX_ENCODE_QUEUE];
    std::vector<CNvQueue<EncodeBuffer>> m_EncodeBufferQueue;
    std::vector<CNvQueue<MotionEstimationBuffer>> m_MVBufferQueue;
    std::vector<EncodeOutputBuffer> m_stEOSOutputBfr; 

protected:
    NVENCSTATUS                                          Deinitialize(uint32_t devicetype);
    NVENCSTATUS EncodeFrame(EncodeFrameConfig *pEncodeFrame, bool bFlush=false, uint32_t width=0, uint32_t height=0, int cam_idx = 0);
    NVENCSTATUS InitCuda(uint32_t deviceID = 0);
    NVENCSTATUS AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight, NV_ENC_BUFFER_FORMAT inputFormat, int num_of_cams);
    NVENCSTATUS ReleaseIOBuffers();
    NVENCSTATUS FlushEncoder(int cam_idx);
    void FlushMVOutputBuffer(int cam_idx);
    
    uint32_t  chromaFormatIDC = 0;
    int8_t *qpDeltaMapArray = NULL;
    uint32_t qpDeltaMapArraySize = 0;
    FILE *fpExternalHint = NULL;
    NVENC_EXTERNAL_ME_HINT *ceaBuffer = NULL;
    bool doNotReleaseContext = false;
    int num_cameras = 1;
};

// NVEncodeAPI entry point
typedef NVENCSTATUS (NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST*); 

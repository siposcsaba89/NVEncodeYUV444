#include "NvEncoder/cuviddec.h"
#include "NvEncoder/nvcuvid.h"
#include <Windows.h>

//TODO: danger, not fail safe cannot check if succeded or not, use init function instead
HINSTANCE hGetProcIDDLL = LoadLibrary("nvcuvid.dll");

typedef CUresult(*cuvidCreateDecoder_type)(CUvideodecoder * phDecoder, CUVIDDECODECREATEINFO * pdci);
typedef CUresult(*cuvidCreateVideoParser_type)(CUvideoparser * pObj, CUVIDPARSERPARAMS * pParams);
typedef CUresult(*cuvidCtxLockCreate_type)(CUvideoctxlock * pLock, CUcontext ctx);
typedef CUresult(*cuvidCtxLockDestroy_type)(CUvideoctxlock lck);
typedef CUresult(*cuvidDecodePicture_type)(CUvideodecoder hDecoder, CUVIDPICPARAMS * pPicParams);
typedef CUresult(*cuvidDestroyDecoder_type)(CUvideodecoder hDecoder);
typedef CUresult(*cuvidDestroyVideoParser_type)(CUvideoparser obj);
typedef CUresult(*cuvidGetDecoderCaps_type)(CUVIDDECODECAPS * pdc);
typedef CUresult(*cuvidGetDecodeStatus_type)(CUvideodecoder hDecoder, int nPicIdx, CUVIDGETDECODESTATUS * pDecodeStatus);
typedef CUresult(*cuvidMapVideoFrame64_type)(CUvideodecoder hDecoder, int nPicIdx, unsigned long long * pDevPtr, unsigned int * pPitch, CUVIDPROCPARAMS * pVPP);
typedef CUresult(*cuvidParseVideoData_type)(CUvideoparser obj, CUVIDSOURCEDATAPACKET * pPacket);
typedef CUresult(*cuvidReconfigureDecoder_type)(CUvideodecoder hDecoder, CUVIDRECONFIGUREDECODERINFO * pDecReconfigParams);
typedef CUresult (*cuvidUnmapVideoFrame64_type)(CUvideodecoder hDecoder, unsigned long long DevPtr);

cuvidCreateDecoder_type cuvidCreateDecoder_ptr = (cuvidCreateDecoder_type)GetProcAddress(hGetProcIDDLL, "cuvidCreateDecoder");
cuvidCreateVideoParser_type cuvidCreateVideoParser_ptr = (cuvidCreateVideoParser_type)GetProcAddress(hGetProcIDDLL, "cuvidCreateVideoParser");
cuvidCtxLockCreate_type cuvidCtxLockCreate_ptr = (cuvidCtxLockCreate_type)GetProcAddress(hGetProcIDDLL, "cuvidCtxLockCreate");
cuvidCtxLockDestroy_type cuvidCtxLockDestroy_ptr = (cuvidCtxLockDestroy_type)GetProcAddress(hGetProcIDDLL, "cuvidCtxLockDestroy");
cuvidDecodePicture_type cuvidDecodePicture_ptr = (cuvidDecodePicture_type)GetProcAddress(hGetProcIDDLL, "cuvidDecodePicture");
cuvidDestroyDecoder_type cuvidDestroyDecoder_ptr = (cuvidDestroyDecoder_type)GetProcAddress(hGetProcIDDLL, "cuvidDestroyDecoder");
cuvidDestroyVideoParser_type cuvidDestroyVideoParser_ptr = (cuvidDestroyVideoParser_type)GetProcAddress(hGetProcIDDLL, "cuvidDestroyVideoParser");
cuvidGetDecoderCaps_type cuvidGetDecoderCaps_ptr = (cuvidGetDecoderCaps_type)GetProcAddress(hGetProcIDDLL, "cuvidGetDecoderCaps");
cuvidGetDecodeStatus_type cuvidGetDecodeStatus_ptr = (cuvidGetDecodeStatus_type)GetProcAddress(hGetProcIDDLL, "cuvidGetDecodeStatus");
cuvidMapVideoFrame64_type cuvidMapVideoFrame64_ptr = (cuvidMapVideoFrame64_type)GetProcAddress(hGetProcIDDLL, "cuvidMapVideoFrame64");
cuvidParseVideoData_type cuvidParseVideoData_ptr = (cuvidParseVideoData_type)GetProcAddress(hGetProcIDDLL, "cuvidParseVideoData");
cuvidReconfigureDecoder_type cuvidReconfigureDecoder_ptr = (cuvidReconfigureDecoder_type)GetProcAddress(hGetProcIDDLL, "cuvidReconfigureDecoder");
cuvidUnmapVideoFrame64_type cuvidUnmapVideoFrame64_ptr = (cuvidUnmapVideoFrame64_type)GetProcAddress(hGetProcIDDLL, "cuvidUnmapVideoFrame64");


CUresult CUDAAPI cuvidGetDecoderCaps(CUVIDDECODECAPS * pdc)
{
    return cuvidGetDecoderCaps_ptr(pdc);
}

CUresult CUDAAPI cuvidCreateDecoder(CUvideodecoder * phDecoder, CUVIDDECODECREATEINFO * pdci)
{
    return cuvidCreateDecoder_ptr(phDecoder, pdci);
}

CUresult CUDAAPI cuvidDestroyDecoder(CUvideodecoder hDecoder)
{
    return cuvidDestroyDecoder_ptr(hDecoder);
}

CUresult CUDAAPI cuvidDecodePicture(CUvideodecoder hDecoder, CUVIDPICPARAMS * pPicParams)
{
    return cuvidDecodePicture_ptr(hDecoder, pPicParams);
}

CUresult CUDAAPI cuvidGetDecodeStatus(CUvideodecoder hDecoder, int nPicIdx, CUVIDGETDECODESTATUS * pDecodeStatus)
{
    return cuvidGetDecodeStatus_ptr(hDecoder, nPicIdx, pDecodeStatus);
}

CUresult CUDAAPI cuvidReconfigureDecoder(CUvideodecoder hDecoder, CUVIDRECONFIGUREDECODERINFO * pDecReconfigParams)
{
    return cuvidReconfigureDecoder_ptr(hDecoder, pDecReconfigParams);
}

CUresult CUDAAPI cuvidMapVideoFrame64(CUvideodecoder hDecoder, int nPicIdx, unsigned long long * pDevPtr, unsigned int * pPitch, CUVIDPROCPARAMS * pVPP)
{
    return cuvidMapVideoFrame64_ptr(hDecoder, nPicIdx, pDevPtr, pPitch, pVPP);
}

CUresult CUDAAPI cuvidUnmapVideoFrame64(CUvideodecoder hDecoder, unsigned long long DevPtr)
{
    return cuvidUnmapVideoFrame64_ptr(hDecoder, DevPtr);
}


CUresult CUDAAPI cuvidCreateVideoParser(CUvideoparser * pObj, CUVIDPARSERPARAMS * pParams)
{
    return cuvidCreateVideoParser_ptr(pObj, pParams);
}

CUresult CUDAAPI cuvidParseVideoData(CUvideoparser obj, CUVIDSOURCEDATAPACKET * pPacket)
{
    return cuvidParseVideoData_ptr(obj, pPacket);
}

CUresult CUDAAPI cuvidDestroyVideoParser(CUvideoparser obj)
{
    return cuvidDestroyVideoParser_ptr(obj);
}

CUresult CUDAAPI cuvidCtxLockCreate(CUvideoctxlock * pLock, CUcontext ctx)
{
    return cuvidCtxLockCreate_ptr(pLock, ctx);
}

CUresult CUDAAPI cuvidCtxLockDestroy(CUvideoctxlock lck)
{
    return cuvidCtxLockDestroy_ptr(lck);
}

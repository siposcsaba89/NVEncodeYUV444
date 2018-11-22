#include <iostream>

#include <cuda.h>
#include <NvEncoder/FFmpegDemuxer.h>
#include <opencv2/opencv.hpp>
#include <NvEncoder/NvDecoder.h>

#include <fstream>
#include <ctime>

using namespace std;


void convertBayerToRGBANDG(const cv::Mat & bayer_img, cv::Mat & rgb, cv::Mat & ggg)
{
    rgb = cv::Mat(bayer_img.rows / 2, bayer_img.cols / 2, CV_8UC3);
    ggg = cv::Mat(bayer_img.rows/2, bayer_img.cols / 2, CV_8UC1);

    for (int j = 0; j < bayer_img.rows / 2; ++j)
    {
        for (int i = 0; i < bayer_img.cols / 2; ++i)
        {
            int idx1 = bayer_img.cols * (j * 2 + 0) + i * 2 + 0;
            int idx2 = bayer_img.cols * (j * 2 + 0) + i * 2 + 1;
            int idx3 = bayer_img.cols * (j * 2 + 1) + i * 2 + 0;
            int idx4 = bayer_img.cols * (j * 2 + 1) + i * 2 + 1;
            uint8_t r = bayer_img.data[idx1];
            uint8_t g = bayer_img.data[idx2];
            uint8_t g2 = bayer_img.data[idx3];
            uint8_t b = bayer_img.data[idx4];

            rgb.at<cv::Vec3b>(j, i) = cv::Vec3b(b, g, r);
            ggg.at<uint8_t>(j, i) = g2;
        }
    }

}

void convertRGB2yuv44(const cv::Mat & bgr, cv::Mat & yuv444)
{
    yuv444 = cv::Mat(bgr.rows, bgr.cols, CV_8UC3);
    for (int j = 0; j < bgr.rows; ++j)
    {
        for (int i = 0; i < bgr.cols; ++i)
        {
            cv::Vec3b pix = bgr.at<cv::Vec3b>(j, i);
            float y = 0.257f * pix[2] + 0.504f * pix[1] + 0.098f * pix[0] + 16;
            float u = -0.148f * pix[2] - 0.291f * pix[1] + 0.439f * pix[0] + 128;
            float v = 0.439f * pix[2] - 0.368f * pix[1] - 0.071f * pix[0] + 128;
            yuv444.at<cv::Vec3b>(j, i) = cv::Vec3b(uchar(y), uchar(u), uchar(v));
        }
    }
}

int main()
{
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    CUresult cures = cuCtxCreate(&ctx, 0, dev);
    

    FFmpegDemuxer*  demuxer = new FFmpegDemuxer("d:/tmp/saic/0011/stream02Rec-M_FISHEYE_L_nvidia.-2016-05-06_00-04-22.h264");
    NvDecoder *  decoders = new NvDecoder(
        ctx, demuxer->GetWidth(), demuxer->GetHeight(),
                      false, FFmpeg2NvCodecId(demuxer->GetVideoCodec()), NULL, false, false, NULL, NULL);
    delete decoders;
    delete demuxer;
    cuCtxDestroy(ctx);

    return 0;
}
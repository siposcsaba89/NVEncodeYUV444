#include <iostream>

#include <opencv2/opencv.hpp>
#include <NvEncoder/NvEncoder.h>

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
            yuv444.at<cv::Vec3b>(j, i) = cv::Vec3b(y, u, v);
        }
    }
}

int main()
{
    const string fp = "f:/tmp/20170413/stream00Rec-bas.21959995-2017-04-13_16-56-51.bin";

    const string converted = "d:/outRGB444.h264";

    cv::VideoCapture cap(converted);
    cap.set(CV_CAP_PROP_CONVERT_RGB, 0);
    int wi = int(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int he = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    ifstream f_left(fp, ifstream::binary);
  

    if (!f_left.is_open())
        //if (f_left == nullptr)
    {
        printf("Cannot open file %s!", fp.c_str());
        exit(EXIT_FAILURE);
    }

#define HAVE_HEADER 1
#if HAVE_HEADER
#pragma pack(push, 1)
    struct header
    {
        int32_t width;
        int32_t height;
        int32_t format;
        int32_t compession;
        int32_t metaId;
        char data[100];
    } header_;
#pragma pack(pop)


    f_left.read((char*)&header_.width, sizeof(header_.width));
    f_left.read((char*)&header_.height, sizeof(header_.height));
    f_left.read((char*)&header_.format, sizeof(header_.format));
    f_left.read((char*)&header_.compession, sizeof(header_.compession));
    f_left.read(header_.data, 100);
    f_left.read((char*)&header_.metaId, sizeof(header_.metaId));
    int w = header_.width;
    int h = header_.height;

#else
    int w = 1280;
    int h = 720;

#endif
    int start_from_idx = 0;

    if (start_from_idx > 0)
    {
        size_t h_size = 0;
#if HAVE_HEADER
        h_size = sizeof(header);
#endif
        int64_t offset = (int64_t)h_size + int64_t(start_from_idx) * int64_t(sizeof(double)) + int64_t(start_from_idx) * uint64_t(w) * int64_t(h);

        streampos pos = offset;
        f_left.seekg(0, f_left.end);
        int64_t length = f_left.tellg();
        f_left.seekg(0, f_left.beg);

        f_left.seekg(pos);
    }
    cv::Mat left_raw(h, w, CV_8UC1), rgb, ggg;
    //ofstream out_yuv("test.yuv", ios::binary);
    //ofstream out_g_yuv("testG.yuv", ios::binary);
    int fc = 0;

    
    CNvEncoder encoder;
    encoder.encodeConfig.encoderPreset = "lossless";
    encoder.encodeConfig.codec = 0;
    encoder.encodeConfig.fps = 25;
    encoder.encodeConfig.inputFormat = NV_ENC_BUFFER_FORMAT_YUV444; //yuv444
    encoder.init(w / 2, h / 2, 
    { "d:/myrgb.h264", "d:/myrgb2.h264" }, "");
    //encoder.
    CNvEncoder encoder2;
    encoder2.encodeConfig.encoderPreset = "lossless";
    encoder2.encodeConfig.codec = 0;
    encoder2.encodeConfig.fps = 25;
    encoder2.encodeConfig.inputFormat = NV_ENC_BUFFER_FORMAT_NV12; //yuv444
    encoder2.init(w / 2, h / 2, 
    { "d:/mygreen.h264", "d:/mygreen2.h264" }, "");

    //CNvEncoder encoder3;
    //encoder3.encodeConfig.encoderPreset = "lossless";
    //encoder3.encodeConfig.codec = 0;
    //encoder3.encodeConfig.fps = 25;
    //encoder3.encodeConfig.inputFormat = NV_ENC_BUFFER_FORMAT_NV12; //yuv444
    //encoder3.init(w / 2, h / 2, "d:/mygreen2.h264", "");
    while (!f_left.eof())

    {
        double ts = 0.0;
        //size_t readed = fread(&ts, 1, sizeof(double), f_left);
        //readed = fread(left_raw.data, 1, w * h, f_left);
        f_left.read((char*)&ts, sizeof(double));
        f_left.read((char*)left_raw.data, w * h);
        cv::imshow("input", left_raw);

        convertBayerToRGBANDG(left_raw, rgb, ggg);
        cv::Mat yuv420, g_yuv420, rgb_back;
        convertRGB2yuv44(rgb, yuv420);
        //cv::cvtColor(rgb, yuv420, CV_BGR2YCrCb);
        g_yuv420 = cv::Mat::zeros(540, 640, CV_8UC1);
        memcpy(g_yuv420.data, ggg.data, ggg.rows * ggg.cols);
        vector<cv::Mat> splitted;
        vector<cv::Mat> splitted2;
        cv::split(rgb, splitted);
        cv::split(yuv420, splitted2);
        //out_yuv.write((const char *)splitted[0].data, yuv420.rows * yuv420.cols);
        //out_yuv.write((const char *)splitted[1].data, yuv420.rows * yuv420.cols);
        //out_yuv.write((const char *)splitted[2].data, yuv420.rows * yuv420.cols);
        clock_t t = clock();
        encoder.encodeFrame(splitted[0].data, splitted[1].data, splitted[2].data, 0, fc);
        encoder2.encodeFrame(g_yuv420.data, g_yuv420.data + yuv420.rows * yuv420.cols,
            g_yuv420.data + yuv420.rows * yuv420.cols + yuv420.rows * yuv420.cols / 4, 0, fc);
        encoder.encodeFrame(splitted2[0].data, splitted2[1].data, splitted2[2].data, 1, fc);
        encoder2.encodeFrame(g_yuv420.data, g_yuv420.data + yuv420.rows * yuv420.cols,
            g_yuv420.data + yuv420.rows * yuv420.cols + yuv420.rows * yuv420.cols / 4, 1, fc);
        //encoder3.encodeFrame(g_yuv420.data, g_yuv420.data + yuv420.rows * yuv420.cols,
        //    g_yuv420.data + yuv420.rows * yuv420.cols + yuv420.rows * yuv420.cols / 4, fc);
        cout << "Time: " << clock() - t << "\n";
        //cv::cvtColor(yuv420, rgb_back, CV_YCrCb2BGR);
        //out_g_yuv.write((const char *)g_yuv420.data, g_yuv420.rows * g_yuv420.cols);


        cv::imshow("yuv420", yuv420);
        //cv::imshow("rgb_back", rgb_back);

        //cout << cv::norm(rgb_back - rgb) << endl;

        cv::imshow("bgr", rgb);
        cv::imshow("ggg", ggg);
        ++fc;
        //cv::Mat decoded;
        //cap >> decoded;
        //cout << cv::norm(decoded - rgb) << endl;
        ////cv::cvtColor(decoded, decoded
        //cv::imshow("decoded", decoded);

        int k = cv::waitKey(1);
        if (k == 27)
            break;
    }
    cout << "Frame saved: " << fc << endl;

    return 0;
}
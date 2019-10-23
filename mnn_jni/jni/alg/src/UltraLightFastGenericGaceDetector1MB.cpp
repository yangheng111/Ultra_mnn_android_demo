#include <algorithm>
#include <map>
#include "UltraLightFastGenericGaceDetector1MB.h"

#include <iostream>
#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include <sys/stat.h>

UltraLightFastGenericGaceDetector1MB::UltraLightFastGenericGaceDetector1MB()
{ }

UltraLightFastGenericGaceDetector1MB::~UltraLightFastGenericGaceDetector1MB()
{ }

void UltraLightFastGenericGaceDetector1MB::load(const std::string &model_path, int num_thread)
{
    std::vector<std::string> tmpp = { model_path + "/Mb_Tiny_RFB_FD_train_input_320.mnn" };
    net.load_param(tmpp, num_thread);
}


void UltraLightFastGenericGaceDetector1MB::detectInternal(cv::Mat& img_, std::vector<cv::Rect> &faces)
{
    faces.clear();
    img = img_;
    net.Ultra_infer_img(img,conf_threshold, nms_threshold,OUTPUT_NUM, center_variance, size_variance, anchors,faces);
    return;
}

void UltraLightFastGenericGaceDetector1MB::detect(const cv::Mat& img_, std::vector<cv::Rect> &faces)
{   
    if (img_.empty()) return;

    cv::Mat testImg;
    testImg = img_.t();

    // resize and normal
    cv::resize(img_, testImg, cv::Size(320,240), 0, 0);
    testImg.convertTo(testImg, CV_32FC3);
    testImg = (testImg - img_mean) / img_std;
    
    detectInternal(testImg, faces);

    return;
}
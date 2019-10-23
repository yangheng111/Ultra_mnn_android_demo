#ifndef _NET_H_
#define _NET_H_

#include <vector>
#include <string>
#include <ImageProcess.hpp>
#include <Interpreter.hpp>
#include <Tensor.hpp>
#include <memory>

#include "opencv2/opencv.hpp"
/*
	获取网络输出, 保存成cv::Mat
*/
class Inference_engine_tensor
{
public:
    Inference_engine_tensor()
    { }

    ~Inference_engine_tensor()
    { }

    void add_name(std::string &layer)
    {
        layer_name.push_back(layer);
    }

    float* score(int idx)
    {
        return (float*)out_feat[idx].data;
    }

public:
    std::vector<std::string> layer_name;
    std::vector<cv::Mat> out_feat;
};

class Inference_engine
{
public:
    Inference_engine();
    ~Inference_engine();
	
	// 加载网络参数 Session初始化
    int load_param(std::vector<std::string> &file, int num_thread = 1);
    // 输入预处理设置
	int set_params(int inType, int outType, std::vector<float> &mean, std::vector<float> &scale);
    
	// 网络infer, 输入cv::Mat 输出vector<Mat>, 支持多输出
	int infer_img(cv::Mat& imgs, Inference_engine_tensor& out);
	int infer_imgs(std::vector<cv::Mat>& imgs, std::vector<Inference_engine_tensor>& out);
    int Ultra_infer_img(cv::Mat& img,float conf_threshold,float nms_threshold,int OUTPUT_NUM,float center_variance,float size_variance,float anchors[4][4420],std::vector<cv::Rect> &faces);
	
private: 
    MNN::Interpreter* netPtr;
	MNN::Session* sessionPtr;
    MNN::CV::ImageProcess::Config config;
};
#endif
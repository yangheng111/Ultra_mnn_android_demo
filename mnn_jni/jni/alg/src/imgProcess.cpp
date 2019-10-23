#include "imgProcess.h"
#include "UltraLightFastGenericGaceDetector1MB.h"
#include "opencv2/opencv.hpp"

int detFace(cv::Mat input_rgb, const char* path, std::vector<cv::Rect> &faces)
{
    static UltraLightFastGenericGaceDetector1MB ultralightfastgenericgacedetector1MB;
    static bool is_model_prepared = false;
    if (false == is_model_prepared)
    {
        const char* model_path = path;
        ultralightfastgenericgacedetector1MB.load(model_path);
        is_model_prepared = true;
    }
    ultralightfastgenericgacedetector1MB.detect(input_rgb, faces);

    return 0;
}


//  camera input
int deal(int* pixels, int h, int w, const char* path)
{
	cv::Mat frame(h, w, CV_8UC4, pixels);
    cv::flip(frame, frame, 1);

    cv::Mat input_rgb;
    cv::cvtColor(frame, input_rgb, cv::COLOR_BGRA2RGB);
	input_rgb = input_rgb.t();
	std::vector<cv::Rect> faces;
	detFace(input_rgb, path, faces);

    for (cv::Rect face: faces) 
    {
        cv::Rect vis_box;
        vis_box.x = (int) ((face.x/320.0)*frame.cols);
        vis_box.y = (int) ((face.y/240.0)*frame.rows);
        vis_box.width  = (int) ((face.width/320.0)*frame.cols);
        vis_box.height = (int) ((face.height/240.0)*frame.rows);
        int xmax = vis_box.x+vis_box.width;
        int ymax = vis_box.y+vis_box.height;
        
        cv::rectangle(frame, vis_box, cv::Scalar(255,0,0,255), 1);
    }

    return 0;
}
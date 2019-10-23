#include "net.h"

Inference_engine::Inference_engine()
{ }

Inference_engine::~Inference_engine()
{ 
    if ( netPtr != NULL )
	{
		if ( sessionPtr != NULL)
		{
			netPtr->releaseSession(sessionPtr);
			sessionPtr = NULL;
		}

		delete netPtr;
		netPtr = NULL;
	}
}

int Inference_engine::load_param(std::vector<std::string>& file, int num_thread)
{
    if (!file.empty())
    {
        if (file[0].find(".mnn") != std::string::npos)
        {
	        netPtr = MNN::Interpreter::createFromFile(file[0].c_str());
            if (nullptr == netPtr) return -1;
			
			//Ĭ������MNNForwardType ΪCPU
            MNN::ScheduleConfig sch_config;
            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;
            if ( num_thread > 0 )sch_config.numThread = num_thread;
            sessionPtr = netPtr->createSession(sch_config);
            if (nullptr == sessionPtr) return -1;
        }
        else
        {
            return -1;
        }
    }

    return 0;
}

int Inference_engine::set_params(int srcType, int dstType, 
                                 std::vector<float>& mean, std::vector<float>& scale)
{
    config.destFormat   = (MNN::CV::ImageFormat)dstType;
    config.sourceFormat = (MNN::CV::ImageFormat)srcType;

    // mean��normal
    ::memcpy(config.mean,   &mean[0],   3 * sizeof(float));
    ::memcpy(config.normal, &scale[0],  3 * sizeof(float));

    // filterType��wrap
    config.filterType = (MNN::CV::Filter)(1);
    config.wrap = (MNN::CV::Wrap)(2);

    return 0;
}

// infer
int Inference_engine::infer_img(cv::Mat& img, Inference_engine_tensor& out)
{
	// ��������Ԥ����
    MNN::Tensor* tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);

    // auto resize for full conv network.
    bool auto_resize = false;
    if ( !auto_resize )
    {
        std::vector<int>dims = { 1, img.channels(), img.rows, img.cols };
        netPtr->resizeTensor(tensorPtr, dims);
        netPtr->resizeSession(sessionPtr);
    }

    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    process->convert((const unsigned char*)img.data, img.cols, img.rows, img.step[0], tensorPtr);
    
	// �������
	netPtr->runSession(sessionPtr);

	// ��ȡ�������
    for (int i = 0; i < out.layer_name.size(); i++)
    {
        const char* layer_name = NULL;
        if( strcmp(out.layer_name[i].c_str(), "") != 0)
        {
            layer_name = out.layer_name[i].c_str();
        }
        MNN::Tensor* tensorOutPtr = netPtr->getSessionOutput(sessionPtr, layer_name);

        std::vector<int> shape = tensorOutPtr->shape();
        cv::Mat feat(shape.size(), &shape[0], CV_32F);

        auto tensor = reinterpret_cast<MNN::Tensor*>(tensorOutPtr);
        float *destPtr = (float*)feat.data;
        if (nullptr == destPtr)
        {
            std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), false));
            return hostTensor->elementSize();
        }

        std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), true));
        tensor->copyToHostTensor(hostTensor.get());
        tensor = hostTensor.get();

        auto size = tensor->elementSize();
        ::memcpy(destPtr, tensor->host<float>(), size * sizeof(float));

        out.out_feat.push_back(feat.clone());
    }

    return 0;
}

int Inference_engine::infer_imgs(std::vector<cv::Mat>& imgs, std::vector<Inference_engine_tensor>& out)
{
    for (int i = 0; i < imgs.size(); i++)
    {
        infer_img(imgs[i], out[i]);
    }

    return 0;
}

float iou(cv::Rect box0, cv::Rect box1) 
{
    float xmin0 = box0.x;
    float ymin0 = box0.y;
    float xmax0 = box0.x + box0.width;
    float ymax0 = box0.y + box0.height;
    
    float xmin1 = box1.x;
    float ymin1 = box1.y;
    float xmax1 = box1.x + box1.width;
    float ymax1 = box1.y + box1.height;

    float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
    
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    
    if (u <= 0.0) return 0.0f;
    else          return i/u;
}

int Inference_engine::Ultra_infer_img(cv::Mat& img,float conf_threshold,float nms_threshold,int OUTPUT_NUM,float center_variance,float size_variance,float anchors[4][4420],std::vector<cv::Rect> &faces)
{
    MNN::Tensor* tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);
    // auto resize for full conv network.
    // std::vector<int>dims = { 1, img.channels(), img.rows, img.cols };
    // netPtr->resizeTensor(tensorPtr, dims);
    // netPtr->resizeSession(sessionPtr);
    // std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    // process->convert((const unsigned char*)img.data, img.cols, img.rows, 0, tensorPtr);

    std::vector<int> dims{1, img.rows, img.cols, img.channels()};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, img.data, nhwc_size);
    tensorPtr->copyFromHostTensor(nhwc_Tensor);

	// �������
	netPtr->runSession(sessionPtr);

    // get scores
    char* score_layer_name ="scores";
    MNN::Tensor* tensor_scores = netPtr->getSessionOutput(sessionPtr, score_layer_name);
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto scores_dataPtr = tensor_scores_host.host<float>();

    // get landmarks
    char* landmark_layer_name ="460";
    MNN::Tensor* tensor_boxes = netPtr->getSessionOutput(sessionPtr, landmark_layer_name);
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    auto boxes_dataPtr = tensor_boxes_host.host<float>();

    std::vector<cv::Rect> tmp_faces;
    for(int i =0;i<OUTPUT_NUM;++i){
        float ycenter =     boxes_dataPtr[i*4 + 0] * center_variance * anchors[2][i] + anchors[0][i];
        float xcenter =     boxes_dataPtr[i*4 + 1] * center_variance * anchors[3][i] + anchors[1][i];
        float h       = exp(boxes_dataPtr[i*4 + 2] * size_variance) * anchors[2][i];
        float w       = exp(boxes_dataPtr[i*4 + 3] * size_variance) * anchors[3][i];

        float ymin    = ( ycenter - h * 0.5 ) * img.rows;
        float xmin    = ( xcenter - w * 0.5 ) * img.cols;
        float ymax    = ( ycenter + h * 0.5 ) * img.rows;
        float xmax    = ( xcenter + w * 0.5 ) * img.cols;

        float nonface_prob = scores_dataPtr[i*2 + 0];
        float face_prob    = scores_dataPtr[i*2 + 1];

        if (face_prob > conf_threshold) {
            cv::Rect tmp_face;
            tmp_face.x = (int)xmin;
            tmp_face.y = (int)ymin;
            tmp_face.width  = (int)(xmax - xmin);
            tmp_face.height = (int)(ymax - ymin);
            tmp_faces.push_back(tmp_face);

        }
    }
    // perform NMS
    int N = tmp_faces.size();
    std::vector<int> labels(N, -1); 
    for(int i = 0; i < N-1; ++i)
    {
        for (int j = i+1; j < N; ++j)
        {
            cv::Rect pre_box = tmp_faces[i];
            cv::Rect cur_box = tmp_faces[j];
            float iou_ = iou(pre_box, cur_box);
            if (iou_ > nms_threshold) {
                labels[j] = 0;
            }
        }
    }
    for (int i = 0; i < N; ++i)
    {
        if (labels[i] == -1)
            faces.push_back(tmp_faces[i]);
    }

    return 0;
}
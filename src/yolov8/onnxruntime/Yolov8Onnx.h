#ifndef YOLOV8ONNX_H
#define YOLOV8ONNX_H

#include<InferenceOnnx.h>
#include <opencv2/opencv.hpp>
#include "Datatype.h"

class Yolov8Onnx : public InferenceOnnx {
public:
    explicit  Yolov8Onnx(const char* model_path, cv::Size img_size) : InferenceOnnx(model_path), _img_size(img_size)
    {
    }

    int getInference(cv::Mat& img, std::vector<DET_RESULT> & results);

    void drawResult(cv::Mat& img, const std::vector<DET_RESULT>& results);

    void postProcessing(float* data, float det_scale, std::vector<DET_RESULT> & results);

private:

    cv::Size _img_size = cv::Size(640,640);

    int _anchorLength = 84;
    float _classThreshold = 0.2f;
    float _nmsThrehold = 0.5f;
public:


};

#endif
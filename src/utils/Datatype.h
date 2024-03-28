#ifndef DATATYPE_H
#define DATATYPE_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

typedef struct det{
    cv::Rect_<float> box;
    int label = 0;
    float confidence = 0.0;
}DET_RESULT;

typedef struct face{
    std::vector<cv::Point> landmark;
    cv::Rect_<float> box;
    float confidence = 0.0;
}FACE_RESULT;

// using FACE_OUTPUT = std::vector<ONE_FACE>;

#endif //DATATYPE_H

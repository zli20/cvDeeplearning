
#ifndef YOLOV8DETSNPE_H
#define YOLOV8DETSNPE_H

#include <string>

#include "Datatype.h"
#include "opencv2/opencv.hpp"
#include "SnpeEngine.h"

class SnpeEngine;

class Yolov8DetSnpe :public SnpeEngine{
public:
    explicit  Yolov8DetSnpe(const std::string& model_path, const cv::Size img_size, const int platform) : SnpeEngine(img_size)
    {
        if (1 < platform) {
            const std::vector<std::string>  outnames{"Concat_271", "Split_227"};
            this->setOutName(outnames);
        }
        else {
            const std::vector<std::string>  outnames{"Concat_271"};
            this->setOutName(outnames);
        }
        // initialization models
        if (init(model_path, platform) != 0) {
            throw std::runtime_error("Faild to init model");
        }
    }

    ~Yolov8DetSnpe() override= default;

    static inline float sigmoid(float x) {
        return 1 / (1 + exp(-x));
    }

    // void Preprocessing(cv::Mat &img);
    void preProcessing(cv::Mat &img, float & det_scale) const;
    void postProcessing(std::vector<DET_RESULT> & _results, float det_scale);

    void getInference(cv::Mat& img, std::vector<DET_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<DET_RESULT>& results);

    cv::Size img_input_size;
    const float target_conf_th = 0.5;
    const float nms_th = 0.2;
    const int out_nums = 84;
};


#endif //YOLOV8DETSNPE_H

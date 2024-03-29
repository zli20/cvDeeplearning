#ifndef YOLOV8FACESNPE_H
#define YOLOV8FACESNPE_H

#include <SnpeEngine.h>
#include <string>
#include "Datatype.h"

class Yolov8FaceSnpe :public SnpeEngine{
    public:
    explicit  Yolov8FaceSnpe(const std::string& model_path, const cv::Size img_size, const int platform) : SnpeEngine(img_size)
    {
        const std::vector<std::string>  outnames{"Concat_221", "Concat_229", "Concat_237"};
        this->setOutName(outnames);
        // initialization models
        if (init(model_path, platform) != 0) {
            throw std::runtime_error("Faild to init model");
        }
    }

    ~Yolov8FaceSnpe() override= default;

    // void Preprocessing(cv::Mat &img);
    void preProcessing(cv::Mat &img, float & det_scale) const;

    void postProcessing(std::vector<FACE_RESULT> & _results, float det_scale);

    int getInference(const cv::Mat& img, std::vector<FACE_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const;

    cv::Size img_input_size;
    const float target_conf_th = 0.5;
    const float nms_th = 0.2;

    const int out_nums = 80;
    const int reg_max = 16;
    const int num_point = 5;


};


#endif //YOLOV8FACESNPE_H

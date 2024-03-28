#ifndef YOLOV8FACESNPE_H
#define YOLOV8FACESNPE_H

#include <SnpeEngine.h>
#include <string>
#include "Datatype.h"

class Yolov8FaceSnpe :public SnpeEngine{
    public:
    explicit  Yolov8FaceSnpe(const std::string& model_path, const cv::Size img_size, const int platform) : SnpeEngine(img_size)
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

    ~Yolov8FaceSnpe() override= default;

    static inline float sigmoid(const float x) {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    // void Preprocessing(cv::Mat &img);
    void preProcessing(cv::Mat &img, float & det_scale) const;

    void postProcessing(std::vector<FACE_RESULT> & _results, float det_scale);

    int getInference(const cv::Mat& img, std::vector<FACE_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results);

    void Yolov8FaceSnpe::generate_proposal(
        cv::Mat out, std::vector<cv::Rect>& boxes,
        std::vector<float>& confidences,
        std::vector< std::vector<cv::Point>>& landmarks,
        int imgh,int imgw, float ratioh, float ratiow, int padh, int padw);

    cv::Size img_input_size;
    const float target_conf_th = 0.5;
    const float nms_th = 0.2;
    const int out_nums = 84;

    const int reg_max = 16;
    const int inpWidth = 640;
    const int inpHeight = 640;
    const int num_class = 1;

};



#endif //YOLOV8FACESNPE_H

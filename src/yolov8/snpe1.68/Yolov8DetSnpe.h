
#ifndef YOLOV8DETSNPE_H
#define YOLOV8DETSNPE_H

#include <string>

#include "Datatype.h"
#include "opencv2/opencv.hpp"
#include "SnpeEngine.h"
#include "Maincfg.h"

class Yolov8DetSnpe :public SnpeEngine{
public:

    explicit  Yolov8DetSnpe() : SnpeEngine(Maincfg::instance().model_input_width, Maincfg::instance().model_input_hight)
    {
        this->setOutName(Maincfg::instance().model_output_layer_name);

        // initialization models
        if (init(Maincfg::instance().model_path, Maincfg::instance().runtime) != 0) {
            throw std::runtime_error("Faild to init model");
        }
        target_conf_th = Maincfg::instance().target_conf_th;
        nms_th = Maincfg::instance().nms_th;
    }

    ~Yolov8DetSnpe() override= default;



    static void cvSigmoid(cv::Mat& mat);

    void postProcessing(std::vector<DET_RESULT> & _results, const float det_scale, bool padding);

    void getInference(const cv::Mat& img, std::vector<DET_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<DET_RESULT>& results);

    cv::Size img_input_size;
    float target_conf_th ;
    float nms_th;
    int out_node_nums = 84;
};


#endif //YOLOV8DETSNPE_H

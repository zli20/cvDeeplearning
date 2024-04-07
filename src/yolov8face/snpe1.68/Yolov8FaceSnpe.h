#ifndef YOLOV8FACESNPE_H
#define YOLOV8FACESNPE_H

#include <SnpeEngine.h>
#include <string>
#include "Datatype.h"
#include "Maincfg.h"

class Yolov8FaceSnpe :public SnpeEngine{
    public:
    explicit  Yolov8FaceSnpe() : SnpeEngine(Maincfg::instance().model_input_width, Maincfg::instance().model_input_hight)
    {
        this->setOutName(Maincfg::instance().model_output_layer_name);
        // initialization models
        if (init(Maincfg::instance().model_path, Maincfg::instance().runtime) != 0) {
            throw std::runtime_error("Faild to init model");
        }
        target_conf_th = Maincfg::instance().target_conf_th;
        nms_th = Maincfg::instance().nms_th;
    }

    ~Yolov8FaceSnpe() override= default;

    void postProcessing(std::vector<FACE_RESULT> & _results, float det_scale, bool padding);

    int getInference(const cv::Mat& img, std::vector<FACE_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const;

    cv::Size img_input_size;
    float target_conf_th = 0.5;
    float nms_th = 0.2;

    int out_node_nums = 80;
    int reg_max = 16;
    int box_reg = 64;
    int kps_nums = 5;
};


#endif //YOLOV8FACESNPE_H

#ifndef RETINAFACE_SNPE_H
#define RETINAFACE_SNPE_H

#include <SnpeEngine.h>
#include <string>
#include <iostream>
#include "Datatype.h"
#include "Maincfg.h"

class RetinafaceSnpe : public SnpeEngine{
public:
    using ANCHOR = struct anchor{
        float cx;
        float cy;
        float sx;
        float sy;
    };

    explicit  RetinafaceSnpe() : SnpeEngine(Maincfg::instance().model_input_width, Maincfg::instance().model_input_hight)
    {
        this->setOutName(Maincfg::instance().model_output_layer_name);
        // initialization models
        if (init(Maincfg::instance().model_path, Maincfg::instance().runtime) != 0) {
            throw std::runtime_error("Faild to init model");
        }
        target_conf_th = Maincfg::instance().target_conf_th;
        nms_th = Maincfg::instance().nms_th;
    }

    ~RetinafaceSnpe() override= default;

//    void preProcessing(cv::Mat &img, float & det_scale) const override;

    void postProcessing(std::vector<FACE_RESULT> & _results, float det_scale,  bool padding=false);

    int getInference(const cv::Mat& img, std::vector<FACE_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const;

    static void create_anchor_retinaface(std::vector<ANCHOR> &anchor, int w, int h);

    cv::Size img_input_size;
    float target_conf_th = 0.5;
    float nms_th = 0.2;

    int box_node_nums = 4;
    int cls_node_nums = 2;
    int kps_node_nums = 10;
    int kps_nums = 5;

};



#endif //RETINAFACE_SNPE_H

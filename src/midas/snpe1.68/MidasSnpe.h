#ifndef MIDAS_SNPE_H
#define MIDAS_SNPE_H

#include <SnpeEngine.h>
#include <string>
#include <iostream>
#include "Datatype.h"
#include "Maincfg.h"

class MidasSnpe : public SnpeEngine{
public:

    explicit  MidasSnpe() : SnpeEngine(Maincfg::instance().model_input_width, Maincfg::instance().model_input_hight)
    {
        this->setOutName(Maincfg::instance().model_output_layer_name);
        // initialization models
        if (init(Maincfg::instance().model_path, Maincfg::instance().runtime) != 0) {
            throw std::runtime_error("Faild to init model");
        }
    }

    ~MidasSnpe() override= default;

//    void preProcessing(cv::Mat &img, float & det_scale) const override;

    void postProcessing();

    int getInference(const cv::Mat& img);

    void drawResult(cv::Mat& img) const;

    cv::Size img_input_size;
    cv::Mat result_mat;
    cv::Mat result_mat_resize;
    cv::Mat result_mat_resize_255;
    cv::Mat result_mat_resize_color;

};

#endif // MIDAS_SNPE_H

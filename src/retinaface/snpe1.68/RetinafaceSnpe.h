#ifndef RETINAFACE_SNPE_H
#define RETINAFACE_SNPE_H

#include <SnpeEngine.h>
#include <string>
#include <iostream>
#include "Datatype.h"

class RetinafaceSnpe : public SnpeEngine{
public:
    using ANCHOR = struct anchor{
        float cx;
        float cy;
        float sx;
        float sy;
    };

    explicit  RetinafaceSnpe(const std::string& model_path, const cv::Size img_size, const int platform) : SnpeEngine(img_size)
    {
        const std::vector<std::string>  outnames{"Concat_155", "Concat_205", "Softmax_206"};
        this->setOutName(outnames);
        // initialization models
        if (init(model_path, platform) != 0) {
            throw std::runtime_error("Failed to init model");
        }
    }

    ~RetinafaceSnpe() override= default;

//    void preProcessing(cv::Mat &img, float & det_scale) const override;

    void postProcessing(std::vector<FACE_RESULT> & _results, float det_scale,  bool padding=false);

    int getInference(const cv::Mat& img, std::vector<FACE_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const;

    static void create_anchor_retinaface(std::vector<ANCHOR> &anchor, int w, int h);

    cv::Size img_input_size;
    const float target_conf_th = 0.5;
    const float nms_th = 0.2;

    const int out_nums = 80;
    const int num_point = 5;


};



#endif //RETINAFACE_SNPE_H

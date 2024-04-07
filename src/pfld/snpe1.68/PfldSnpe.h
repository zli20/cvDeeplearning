#ifndef PFLD_SNPE_H
#define PFLD_SNPE_H

#include <SnpeEngine.h>
#include <string>
#include <iostream>
#include "Datatype.h"
#include "Maincfg.h"

class PfldSnpe : public SnpeEngine{
public:

    explicit  PfldSnpe() : SnpeEngine(Maincfg::instance().model_input_width, Maincfg::instance().model_input_hight)
    {
        this->setOutName(Maincfg::instance().model_output_layer_name);
        // initialization models
        if (init(Maincfg::instance().model_path, Maincfg::instance().runtime) != 0) {
            throw std::runtime_error("Faild to init model");
        }
    }

    ~PfldSnpe() override= default;

//    void preProcessing(cv::Mat &img, float & det_scale) const override;

    void postProcessing(std::vector<FACE_RESULT> & _results, int idx, float det_scale, bool padding=false);

    int getInference(const cv::Mat& img, std::vector<FACE_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const;

    static int getMaxface(const std::vector<FACE_RESULT>& results);

    const int kps_nums = 68;

};



#endif // PFLD_SNPE_H

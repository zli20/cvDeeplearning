#ifndef PFLD_SNPE_H
#define PFLD_SNPE_H

#include <SnpeEngine.h>
#include <string>
#include <iostream>
#include "Datatype.h"

class PfldSnpe : public SnpeEngine{
public:

    explicit  PfldSnpe(const std::string& model_path, const cv::Size img_size, const int platform) : SnpeEngine(img_size)
    {
        const std::vector<std::string>  outnames{"fully_connected_0"};
        this->setOutName(outnames);
        // initialization models
        if (init(model_path, platform) != 0) {
            throw std::runtime_error("Failed to init model");
        }
    }

    ~PfldSnpe() override= default;

//    void preProcessing(cv::Mat &img, float & det_scale) const override;

    void postProcessing(std::vector<FACE_RESULT> & _results, int idx, float det_scale, bool padding=false);

    int getInference(const cv::Mat& img, std::vector<FACE_RESULT>& results);

    void drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const;

    static int getMaxface(const std::vector<FACE_RESULT>& results);

    const int num_point = 68;

};



#endif // PFLD_SNPE_H

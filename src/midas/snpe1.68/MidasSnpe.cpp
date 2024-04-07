#include "MidasSnpe.h"
#include "ImageProcessing.h"


void MidasSnpe::postProcessing() {
    for(const auto& it : this->_out_data_ptr) {
        std::cout << it.first << " " << it.second << std::endl;
        auto *pdata = it.second;
        auto shape = this->_output_shapes[it.first];
        result_mat = cv::Mat(cv::Size(shape[1], shape[2]), CV_32F, pdata);

        cv::resize(result_mat, result_mat_resize, cv::Size(img_input_size.width, img_input_size.height), 0, 0, cv::INTER_CUBIC);

        // convert to 255
        double min_val, max_val;
        cv::minMaxLoc(result_mat_resize, &min_val, &max_val);
        result_mat_resize_255 = (result_mat_resize - min_val) / (max_val - min_val) * 255.0;
        result_mat_resize_255.convertTo(result_mat_resize_255, CV_8U);

        // color map
        cv::applyColorMap(result_mat_resize_255, result_mat_resize_color, cv::COLORMAP_INFERNO);
    }
}

void MidasSnpe::drawResult(cv::Mat& img) const {

}

int MidasSnpe::getInference(const cv::Mat &img) {
    this->img_input_size = img.size();
    cv::Mat input_mat(img);

    float det_scale;
    preProcessing(input_mat, det_scale, false, true, true, true);

    build_tensor(input_mat);

    this->inference();

    postProcessing();
    return 0;
}



#include "PfldSnpe.h"
#include "ImageProcessing.h"


void PfldSnpe::postProcessing(std::vector<FACE_RESULT> &_results, int idx, float det_scale, bool padding) {
    float higth_scale = 1.0;
    float width_scale = 1.0;
    if(!padding){
        higth_scale = this->_model_input_height / _results[idx].box.height;
        width_scale = this->_model_input_width / _results[idx].box.width;
    }
    for(const auto & it : this->_out_data_ptr){
        auto *pdata = it.second;
        auto shape= this->_output_shapes[it.first];
        size_t data_size = shape[0] * shape[1];
        for (size_t j = 0; j < data_size /2; j++) {
            auto x = static_cast<float>(pdata[j * 2]) * this->_model_input_width / det_scale / width_scale +  _results[idx].box.x;
            auto y = static_cast<float>(pdata[j * 2 + 1]) * this->_model_input_height / det_scale / higth_scale + +  _results[idx].box.y;
            _results[idx].mulkeypoints.push_back(x);
            _results[idx].mulkeypoints.push_back(y);
        }
    }
}

void PfldSnpe::drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const {
    if (results.empty()) {
        return ;
    }
    for (auto& result : results) {
        int  left, top, width, height;
        left = result.box.x;
        top = result.box.y;
        width = result.box.width;
        height = result.box.height;

        cv::Rect boxxs;
        boxxs.x = left;
        boxxs.y = top;
        boxxs.width = width;
        boxxs.height = height;

        cv::rectangle(img, boxxs, cv::Scalar(0, 0, 255), 1, 8);
        std::string label = std::to_string(result.confidence);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = std::max(top, labelSize.height);
        putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

        if(!result.mulkeypoints.empty()) {
            for (int k = 0; k < this->kps_nums; k++) {
                int kpt_x = std::round(result.mulkeypoints[k * 2]);
                int kpt_y = std::round(result.mulkeypoints[k * 2 + 1]);
                cv::Scalar kps_color = cv::Scalar(0, 255, 255);
                cv::circle(img, {kpt_x, kpt_y}, 2, kps_color, -1);
            }
        }
    }
}

int PfldSnpe::getInference(const cv::Mat &img, std::vector<FACE_RESULT> &results) {
    if(!results.empty()){
        int idx = getMaxface(results);

        int centerX = results[idx].box.x + results[idx].box.width / 2;
        int centerY = results[idx].box.y + results[idx].box.height / 2;

        int inflatedWidth = static_cast<int>(results[idx].box.width * 1.1);
        int inflatedHeight = static_cast<int>(results[idx].box.height * 1.1);

        int newX1 = centerX - inflatedWidth / 2;
        int newY1 = centerY - inflatedHeight / 2;
        int newX2 = centerX + inflatedWidth / 2;
        int newY2 = centerY + inflatedHeight / 2;

        newX1 = std::max(0, newX1);
        newY1 = std::max(0, newY1);
        newX2 = std::min(img.cols, newX2);
        newY2 = std::min(img.rows, newY2);

        std::vector<float> box;
        // 将计算得到的坐标值添加到 box 向量中
        box.push_back(static_cast<float>(newX1));
        box.push_back(static_cast<float>(newX2));
        box.push_back(static_cast<float>(newY1));
        box.push_back(static_cast<float>(newY2));

        cv::Rect inflatedBbox(newX1, newY1, newX2 - newX1, newY2 - newY1);
        cv::Mat input_mat = img(inflatedBbox).clone();
        float det_scale;
        preProcessing(input_mat, det_scale, true, true, true, true);

        this->build_tensor(input_mat);

        this->inference();

        postProcessing(results, idx, det_scale, true);

        return 0;
    }
    return -1;
}

int PfldSnpe::getMaxface(const std::vector<FACE_RESULT> &results) {
    int _idx = -1;
    float area = 0;
    for (size_t i = 0; i < results.size(); i++) {
        int w = results[i].box.width;
        int h = results[i].box.height;
        if (w * h >= area) {
            area = w * h;
            _idx = i;
        }
    }
    return _idx;
}


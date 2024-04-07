#include "RetinafaceSnpe.h"
#include "ImageProcessing.h"


void RetinafaceSnpe::postProcessing(std::vector<FACE_RESULT> &_results, float det_scale, bool padding) {
    float higth_scale = 1.0;
    float width_scale = 1.0;
    if(!padding){
        higth_scale = 640.0 / this->img_input_size.height;
        width_scale = 640.0 / this->img_input_size.width;
    }

    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels;
    std::vector<std::vector<float>> landmarks;
    auto * boxptr = this->_out_data_ptr[Maincfg::instance().model_output_tensor_name[0]];
    auto * kpsptr = this->_out_data_ptr[Maincfg::instance().model_output_tensor_name[1]];
    auto * clsptr = this->_out_data_ptr[Maincfg::instance().model_output_tensor_name[2]];

    std::vector<ANCHOR> anchors;
    create_anchor_retinaface(anchors, this->_model_input_width,this->_model_input_height);

    for (auto anchor : anchors)
    {
        if (clsptr[1] > this->target_conf_th)
        {
            ANCHOR tmp_anchor;
            tmp_anchor.cx = anchor.cx + boxptr[0] * 0.1 * anchor.sx;
            tmp_anchor.cy = anchor.cy + boxptr[1] * 0.1 * anchor.sy;
            tmp_anchor.sx = anchor.sx * exp(boxptr[2] * 0.2);
            tmp_anchor.sy = anchor.sy * exp(boxptr[3] * 0.2);

            float xmin = std::max((tmp_anchor.cx - tmp_anchor.sx/2) * this->_model_input_width / width_scale / det_scale, 0.f) ;
            float ymin = std::max((tmp_anchor.cy - tmp_anchor.sy/2) * this->_model_input_height / higth_scale / det_scale, 0.f) ;
            float xmax = std::min((tmp_anchor.cx + tmp_anchor.sx/2) * this->_model_input_width / width_scale / det_scale, float(img_input_size.width - 1));
            float ymax = std::min((tmp_anchor.cy + tmp_anchor.sy/2) * this->_model_input_height / higth_scale / det_scale, float(img_input_size.height - 1));
            cv::Rect box = cv::Rect(xmin, ymin, (xmax - xmin), (ymax - ymin));
            float score = clsptr[1];
            std::vector<float>kpts;
            for (int j = 0; j < this->kps_nums; ++j)
            {
                kpts.push_back(((anchor.cx + kpsptr[2 * j] * 0.1 * anchor.sx) * this->_model_input_width) / width_scale / det_scale);
                kpts.push_back(((anchor.cy + kpsptr[2 * j + 1] * 0.1 * anchor.sy) * this->_model_input_height) / width_scale / det_scale);
            }
            boxes.push_back(box);
            confidences.push_back(score);
            labels.push_back(0);
            landmarks.push_back(kpts);
        }
        boxptr += box_node_nums;
        clsptr += cls_node_nums;
        kpsptr += kps_node_nums;
    }

#if 1
//    auto start = static_cast<double>(cv::getTickCount());
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, target_conf_th, nms_th, nms_result);
//    auto end = static_cast<double>(cv::getTickCount());
//    auto time_cost = (end - start) / cv::getTickFrequency() * 1000;
//    std::cout << "---------------------_nms time cost :      " << time_cost << "   ms" << std::endl;

    for (auto idx : nms_result) {
        FACE_RESULT result;
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.label = labels[idx];
        result.landmark = landmarks[idx];
        _results.push_back(result);
    }
#elif
    auto start = static_cast<double>(cv::getTickCount());
    _nms(boxes, confidences, landmarks, labels, nms_th);
    auto end = static_cast<double>(cv::getTickCount());
    auto time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "---------------------_nms time cost :      " << time_cost << "   ms" << std::endl;

    for(size_t  idx = 0; idx < boxes.size(); idx++){
        FACE_RESULT result;
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.label = labels[idx];
        result.landmark = landmarks[idx];
        _results.push_back(result);
    }
#endif

}

void RetinafaceSnpe::drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const {
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

        for (int k = 0; k < this->kps_nums; k++)
        {
            int kpt_x = std::round(result.landmark[k * 2]);
            int kpt_y = std::round(result.landmark[k * 2 + 1]);
            cv::Scalar kps_color = cv::Scalar(0, 255, 255);
            cv::circle(img, { kpt_x, kpt_y }, 3, kps_color, -1);
        }
    }
}

int RetinafaceSnpe::getInference(const cv::Mat &img, std::vector<FACE_RESULT> &results) {
    this->img_input_size = img.size();
    cv::Mat input_mat(img);

    float det_scale;
    preProcessing(input_mat, det_scale, true, false, true, true);

    build_tensor(input_mat);

    this->inference();

    results.clear();
    postProcessing(results, det_scale, true);
    return 0;
}

void RetinafaceSnpe::create_anchor_retinaface(std::vector<ANCHOR> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (size_t  i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (size_t  k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l : min_size)
                {
                    float s_kx = l*1.0/w;
                    float s_ky = l*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    ANCHOR axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

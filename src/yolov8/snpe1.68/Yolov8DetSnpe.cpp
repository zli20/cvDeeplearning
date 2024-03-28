#include "Yolov8DetSnpe.h"
#include "ImageProcessing.h"

void Yolov8DetSnpe::preProcessing(cv::Mat &img, float & det_scale) const{
    det_scale=1.0;
    if(img.size() != model_input_size) {
        // resize_padding(img, det_scale, this->img_input_size);
        cv::resize(img, img, cv::Size(model_input_hight, model_input_width));
    }
    // cv::imshow("YOLOv8: ", img);
    // cv::waitKey();
    img.convertTo(img, CV_32F, 1.0/255);
    cv::Scalar meanValues(103.0 / 255, 117.0 / 255, 123.0 / 255);
    img -= meanValues;
    // BGR to RGB
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

}

void Yolov8DetSnpe::postProcessing(std::vector<DET_RESULT>& _results, float det_scale) {
    float higth_scale = 640.0 / this->img_input_size.height;
    float width_scale = 640.0 / this->img_input_size.width;

    if(this->platform == DSP) {

        for (const auto& pair : _out_data_ptr) {
            std::cout << "Key: " << pair.first << ", Value: " << *(pair.second) << std::endl;
        }
        auto *boxes_data = this->_out_data_ptr["output0"];
        auto *scores_data = this->_out_data_ptr["382"];

        const cv::Mat boxes_mat = cv::Mat(cv::Size(8400, 84), CV_32F, boxes_data).t();
        cv::Mat scores_mat = cv::Mat(cv::Size(8400, 80), CV_32F, scores_data).t();
        cvSigmoid(scores_mat);


        auto boxes_pdata = reinterpret_cast<float*>(boxes_mat.data);
        auto scores_pdata = reinterpret_cast<float*>(scores_mat.data);

        const int rows = boxes_mat.rows;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<int> labels;
        const int socre_array_length = out_nums - 4;
        for (int r = 0; r < rows; ++r) {
            cv::Mat scores(1, socre_array_length, CV_32FC1, scores_pdata);

            float box_x = boxes_pdata[0] / width_scale; //x
            float box_y = boxes_pdata[1] / higth_scale; //y
            float box_w = boxes_pdata[2] / width_scale; //w
            float box_h = boxes_pdata[3] / higth_scale; //h

            cv::Point classIdPoint;
            double max_class_socre;
            cv::minMaxLoc(scores, nullptr, &max_class_socre, nullptr, &classIdPoint);
            max_class_socre = static_cast<float>(max_class_socre);
            if (max_class_socre >= target_conf_th) {
                int left = MAX(int(box_x - 0.5 * box_w + 0.5), 0);
                int top = MAX(int(box_y - 0.5 * box_h + 0.5), 0);

                confidences.push_back(max_class_socre);
                labels.push_back(classIdPoint.x);
                boxes.emplace_back(lround(left), lround(top), lround(box_w + 0.5), lround(box_h + 0.5));
            }
            scores_pdata += 80;
            boxes_pdata += 84;
        }

        auto start = std::chrono::system_clock::now();
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, target_conf_th, nms_th, nms_result);
        auto end = std::chrono::system_clock::now();
        auto detect_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();//ms
        std::cout<<"nms time: "<<detect_time<<std::endl;

        for (auto idx : nms_result) {
            DET_RESULT result;
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            result.label = labels[idx];
            _results.push_back(result);
        }
    }
    else {
        auto *data = this->_out_data_ptr["output0"];
        const cv::Mat tensor_mat = cv::Mat(cv::Size(8400, 84), CV_32F, data).t();
        auto pdata = (float*)tensor_mat.data;

        const int rows = tensor_mat.rows;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<int> labels;
        const int socre_array_length = out_nums - 4;
        for (int r = 0; r < rows; ++r) {
            cv::Mat scores(1, socre_array_length, CV_32FC1, pdata+4);

            float box_x = pdata[0] / width_scale; //x
            float box_y = pdata[1] / higth_scale; //y
            float box_w = pdata[2] / width_scale; //w
            float box_h = pdata[3] / higth_scale; //h

            // float box_x = pdata[0] / det_scale; //x
            // float box_y = pdata[1] / det_scale; //y
            // float box_w = pdata[2] / det_scale; //w
            // float box_h = pdata[3] / det_scale; //h

            cv::Point classIdPoint;
            double max_class_socre;
            cv::minMaxLoc(scores, nullptr, &max_class_socre, nullptr, &classIdPoint);
            max_class_socre = static_cast<float>(max_class_socre);
            if (max_class_socre >= target_conf_th) {
                int left;
                left = MAX(int(box_x - 0.5 * box_w + 0.5), 0);
                int top = MAX(int(box_y - 0.5 * box_h + 0.5), 0);

                confidences.push_back(max_class_socre);
                labels.push_back(classIdPoint.x);
                boxes.emplace_back(lround(left), lround(top), lround(box_w + 0.5), lround(box_h + 0.5));
            }
            pdata += out_nums;
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, target_conf_th, nms_th, nms_result);

        for (auto idx : nms_result) {
            DET_RESULT result;
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            result.label = labels[idx];
            _results.push_back(result);
        }
    }
}

void Yolov8DetSnpe::getInference(cv::Mat &img, std::vector<DET_RESULT>& results) {
    this->img_input_size = img.size();
    cv::Mat input_mat(img);

    float det_scale;
    preProcessing(input_mat, det_scale);

    build_tensor(input_mat);

    this->inference();

    results.clear();
    postProcessing(results, det_scale);

}

void Yolov8DetSnpe::drawResult(cv::Mat &img, const std::vector<DET_RESULT> & results) {
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

        // 在目标框左上角标识目标类别以及概率
        if (result.label >= 0 && result.label < cocoClassNamesList.size()) {
            std::string label = cocoClassNamesList[result.label] + ":" + std::to_string(result.confidence);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            top = std::max(top, labelSize.height);
            putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
        else {
            // 如果 result.label 不在合法范围内，输出错误信息
            std::cout << "Error: Invalid label index!" << std::endl;
        }
    }
}

#include <Yolov8Onnx.h>

#include "engine/utils/ImageProcessing.h"

int Yolov8Onnx::getInference(cv::Mat& img, std::vector<DET_RESULT> & results)
{
	float det_scale;
	resize_padding(img, det_scale, this->_img_size);
	//img.convertTo(img, CV_8U);
	img = cv::dnn::blobFromImage(img, 1.0/255, cv::Size(), cv::Scalar(103, 117, 123), true, false, CV_32F);
	auto output_tensor = this->Inference(img);

    auto* all_data = output_tensor[0].GetTensorMutableData<float>();
    postProcessing(all_data, det_scale, results);
	return 0;
}

void Yolov8Onnx::drawResult(cv::Mat& img, const std::vector<DET_RESULT>& results)
{
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

        if (result.label >= 0 && result.label < static_cast<int>(cocoClassNamesList.size())) {
            std::string label = cocoClassNamesList[result.label] + ":" + std::to_string(result.confidence);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            top = std::max(top, labelSize.height);
            putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
        else {
            std::cout << "Error: Invalid label index!" << std::endl;
        }

    }
}

void Yolov8Onnx::postProcessing(float* data, float det_scale, std::vector<DET_RESULT> & results) {
    // std::cout << (int)_outputTensorShape[2] << " " << (int)_outputTensorShape[1] << std::endl;
    std::vector<int64_t>_outputTensorShape = this->_output_shapes.begin()->second;

    cv::Mat output0 = cv::Mat(cv::Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, data).t();
    auto* pdata = (float*)output0.data;
    int rows = output0.rows;
    // int net_width = output0.cols;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels;
    int socre_array_length = this->_anchorLength - 4;
    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, socre_array_length, CV_32FC1, pdata + 4);

        float x = pdata[0] / det_scale; //x
        float y = pdata[1] / det_scale; //y
        float w = pdata[2] / det_scale; //w
        float h = pdata[3] / det_scale; //h

        cv::Point classIdPoint;
        double max_class_socre;
        cv::minMaxLoc(scores, nullptr, &max_class_socre, nullptr, &classIdPoint);
        max_class_socre = (float)max_class_socre;
        if (max_class_socre >= _classThreshold) {
            // rect [x,y,w,h]
            //float x = pdata[0] / det_scale; //x
            //float y = pdata[1] / det_scale; //y
            //float w = pdata[2] / det_scale; //w
            //float h = pdata[3] / det_scale; //h

            int left = MAX(int(x - 0.5 * w + 0.5), 0);
            int top = MAX(int(y - 0.5 * h + 0.5), 0);

            confidences.push_back(max_class_socre);
            labels.push_back(classIdPoint.x);
            boxes.emplace_back(left, top, int(w + 0.5), int(h + 0.5));
        }
        pdata += _anchorLength;
    }

    // NMS
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThrehold, nms_result);

    results.clear();
    for (int idx : nms_result) {
        DET_RESULT result;

        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.label = labels[idx];
        results.push_back(result);
    }
}

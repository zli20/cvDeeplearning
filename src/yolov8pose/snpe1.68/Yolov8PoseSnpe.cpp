#include "Yolov8PoseSnpe.h"
#include "ImageProcessing.h"
#include <cmath>


int Yolov8FaceSnpe::getInference(const cv::Mat &img, std::vector<POSE_RESULT>& results) {
	this->img_input_size = img.size();
	cv::Mat input_mat(img);

	float det_scale;
	preProcessing(input_mat, det_scale, true, true, true, true);

	build_tensor(input_mat);

	this->inference();

	results.clear();
    postProcessing(results, det_scale, true);
	return 0;

}

void Yolov8FaceSnpe::postProcessing(std::vector<POSE_RESULT> & _results, float det_scale, bool padding) {
    float higth_scale = 1.0;
    float width_scale = 1.0;
    if(!padding){
        higth_scale = this->_model_input_height / static_cast<float>(this->img_input_size.height);
        width_scale = this->_model_input_width / static_cast<float>(this->img_input_size.width);
    }

	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> labels;
	std::vector<std::vector<float>> kpss;

	for(const auto& it : this->_out_data_ptr) {
//		std::cout << it.first <<" "<< it.second << std::endl;
		auto *data = it.second;
		auto shape= this->_output_shapes[it.first];

		cv::Mat output0 = cv::Mat(cv::Size(shape[2], shape[1]), CV_32F, data).t();
		auto* pdata = (float*)output0.data;
		int rows = output0.rows;
		for (int r = 0; r < rows; ++r) {
			auto kps_ptr = pdata + 5;
			float score = pdata[4];
			if (score > target_conf_th) {
				// rect [x,y,w,h]
				float x = pdata[0] / width_scale / det_scale; //x
				float y = pdata[1] / higth_scale / det_scale; //y
				float w = pdata[2] / width_scale / det_scale; //w
				float h = pdata[3] / higth_scale / det_scale; //h

				int left = MAX(lround(x - 0.5 * w + 0.5), 0);
				int top = MAX(lround(y - 0.5 * h + 0.5), 0);

				std::vector<float> kps;
				for (int k = 0; k < this->kps_nums; k++) {
					float kps_x = (*(kps_ptr + 3 * k) / width_scale / det_scale);
					float kps_y = (*(kps_ptr + 3 * k + 1) / higth_scale / det_scale);
					float kps_s = *(kps_ptr + 3 * k + 2);

					kps.push_back(kps_x);
					kps.push_back(kps_y);
					kps.push_back(kps_s);
				}
				confidences.push_back(score);
				labels.push_back(0);
				kpss.push_back(kps);
				boxes.emplace_back(left, top, lround(w + 0.5), lround(h + 0.5));
			}
			pdata += out_node_nums;
		}
	}

	// NMS处理
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, target_conf_th, nms_th, nms_result);

	for (const auto idx : nms_result) {
		POSE_RESULT result;

		result.confidence = confidences[idx];
		result.box = boxes[idx];
		result.label = labels[idx];
		result.landmark = kpss[idx];
		_results.push_back(result);
	}
}

void Yolov8FaceSnpe::drawResult(cv::Mat& img, const std::vector<POSE_RESULT>& results) const {
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

		for (int k = 0; k < this->kps_nums; k++) {
			int kpt_x = std::round(result.landmark[k * 3]);
			int kpt_y = std::round(result.landmark[k * 3 + 1]);
			float kpt_conf = result.landmark[k * 3 + 2];
			if (kpt_conf > 0.2f) {
				cv::Scalar kps_color = cv::Scalar(0, 255, 255);
				cv::circle(img, { kpt_x, kpt_y }, 3, kps_color, -1);
			}

			auto& ske = SKELETON[k];
			int pos1_x = std::round(result.landmark[(ske[0] - 1) * 3]);
			int pos1_y = std::round(result.landmark[(ske[0] - 1) * 3 + 1]);

			int pos2_x = std::round(result.landmark[(ske[1] - 1) * 3]);
			int pos2_y = std::round(result.landmark[(ske[1] - 1) * 3 + 1]);

			float pos1_s = result.landmark[(ske[0] - 1) * 3 + 2];
			float pos2_s = result.landmark[(ske[1] - 1) * 3 + 2];

			if (pos1_s > 0.2f && pos2_s > 0.2f) {
				cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
				cv::line(img, { pos1_x, pos1_y }, { pos2_x, pos2_y }, limb_color);
			}
		}
	}
}

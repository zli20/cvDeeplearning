#include "Yolov8FaceSnpe.h"
#include "ImageProcessing.h"
#include <cmath>

void Yolov8FaceSnpe::preProcessing(cv::Mat &img, float & det_scale) const{
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

int Yolov8FaceSnpe::getInference(const cv::Mat &img, std::vector<FACE_RESULT>& results) {
	this->img_input_size = img.size();
	cv::Mat input_mat(img);

	float det_scale;
	preProcessing(input_mat, det_scale);

	build_tensor(input_mat);

	this->inference();

	results.clear();
	postProcessing(results, det_scale);
	return 0;
}

void Yolov8FaceSnpe::postProcessing(std::vector<FACE_RESULT> & _results, float det_scale) {
	float higth_scale = 640.0 / this->img_input_size.height;
	float width_scale = 640.0 / this->img_input_size.width;

	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> labels;
	std::vector<std::vector<float>> landmarks;
	for(const auto& it : this->_out_data_ptr){
		std::cout << it.first <<" "<< it.second << std::endl;
		auto *pdata = it.second;
		auto shape= this->_output_shapes[it.first];
		std:: cout << shape[0] << " " << shape[3]<< std::endl;
		const int feat_h = shape[1];
		const int feat_w = shape[2];
		// const int area = feat_h * feat_w;
		const int stride = (int)ceil((float)this->model_input_hight / feat_h);

		for (int i = 0; i < feat_h; i++) {
			for (int j = 0; j < feat_w; j++)
			{
				float conf = pdata[64];
				float box_prob = sigmoid_x(conf);

				if (box_prob > this->target_conf_th) {
					float pred_ltrb[4];
//                    std::unique_ptr<float[]> dfl_value(new float[reg_max]);
//                    std::unique_ptr<float[]> dfl_softmax(new float[reg_max]);
					auto* dfl_value = new float[reg_max];
					auto* dfl_softmax = new float[reg_max];
					for (int k = 0; k < 4; k++) {
						for (int n = 0; n < reg_max; n++)
						{
							dfl_value[n] = pdata[n+k*reg_max];
						}
						softmax_(dfl_value, dfl_softmax, reg_max);
						float dis = 0.f;
						for (int n = 0; n < reg_max; n++)
						{
							dis += n * dfl_softmax[n];
						}
						pred_ltrb[k] = dis * stride;
					}
					float cx = (j + 0.5f)*stride;
					float cy = (i + 0.5f)*stride;
					float xmin = std::max((cx - pred_ltrb[0]) / width_scale, 0.f) ;
					float ymin = std::max((cy - pred_ltrb[1]) / higth_scale, 0.f) ;
					float xmax = std::min((cx + pred_ltrb[2]) / width_scale, float(img_input_size.width - 1));
					float ymax = std::min((cy + pred_ltrb[3]) / higth_scale, float(img_input_size.height - 1));
					cv::Rect box = cv::Rect(xmin, ymin, (xmax - xmin), (ymax - ymin));
					boxes.push_back(box);
					confidences.push_back(box_prob);
					labels.push_back(0);

					std::vector<float>kpts;
					for (int k = 0; k < this->num_point; k++)
					{
						float kpt_x = (pdata[k * 3 + 64 + 1] * 2 + j)*stride / width_scale;
						float kpt_y = (pdata[k * 3 + 64 + 1 + 1] * 2 + i)*stride / higth_scale;
						float kpt_conf = sigmoid_x(pdata[k * 3 + + 64 + 1 + 2]);
						kpts.push_back(kpt_x);
						kpts.push_back(kpt_y);
						kpts.push_back(kpt_conf);
					}
					landmarks.push_back(kpts);

                    delete[] dfl_value;
                    delete[] dfl_softmax;
				}
				pdata += this->out_nums;
			}
		}
	}

	auto start = std::chrono::system_clock::now();
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, target_conf_th, nms_th, nms_result);
	auto end = std::chrono::system_clock::now();
	auto detect_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();//ms
	std::cout<<"nms time: "<<detect_time<<std::endl;

	for (auto idx : nms_result) {
		FACE_RESULT result;
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		result.label = labels[idx];
		result.landmark = landmarks[idx];
		_results.push_back(result);
	}

    for (auto& kpts : landmarks) {
        kpts.clear();
    }
    landmarks.clear();

}

void Yolov8FaceSnpe::drawResult(cv::Mat& img, const std::vector<FACE_RESULT>& results) const {
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

		for (int k = 0; k < this->num_point; k++)
		{
			int kpt_x = std::round(result.landmark[k * 3]);
			int kpt_y = std::round(result.landmark[k * 3 + 1]);
			float kpt_conf = result.landmark[k * 3 + 2];
			if (kpt_conf > 0.2f) {
				cv::Scalar kps_color = cv::Scalar(0, 255, 255);
				cv::circle(img, { kpt_x, kpt_y }, 3, kps_color, -1);
			}
		}
	}
}

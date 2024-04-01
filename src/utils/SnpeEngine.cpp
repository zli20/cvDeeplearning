#include "SnpeEngine.h"
#include <fstream>
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlEnums.hpp"

void SnpeEngine::checkRuntime(){
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; // 打印版本号
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::AIP_FIXED_TF)) {
        const char *aip_runtime_string = zdl::DlSystem::RuntimeList::runtimeToString(zdl::DlSystem::Runtime_t::AIP_FIXED_TF);
        std::cout << "Current SNPE runtime Support :   " << aip_runtime_string << std::endl;
    }
    else {
        std::cout << "Current SNPE runtime Not Support :  AIP" << std::endl;
    }

    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        const char *gpu_runtime_string = zdl::DlSystem::RuntimeList::runtimeToString(zdl::DlSystem::Runtime_t::GPU);
        std::cout << "Current SNPE runtime Support :   " << gpu_runtime_string << std::endl;
    }
    else {
        std::cout << "Current SNPE runtime Not Support :   GPU" << std::endl;
    }

    if(zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
        const char *dsp_runtime_string = zdl::DlSystem::RuntimeList::runtimeToString(zdl::DlSystem::Runtime_t::DSP);
        std::cout << "Current SNPE runtime Support :   " << dsp_runtime_string << std::endl;
    }
    else {
        std::cout << "Current SNPE runtime Not Support :   DSP" << std::endl;
    }
}

void SnpeEngine::setruntime(const int platform) {// 设置推理硬件顺序
    _runtime_list.clear();
    this->platform = static_cast<Platform>(platform);
    switch (platform) {
        case 3:
            _runtime_list.add(zdl::DlSystem::Runtime_t::AIP_FIXED_TF);
        case 2:
            _runtime_list.add(zdl::DlSystem::Runtime_t::DSP);
        case 1:
            _runtime_list.add(zdl::DlSystem::Runtime_t::GPU);
            // break;
        case 0:
        default:
            _runtime_list.add(zdl::DlSystem::Runtime_t::CPU);
            break;
    }
}

void SnpeEngine::setOutName(const std::vector<std::string>& out_names) {
    for (const auto& name : out_names) {
        _output_tensor_names.append(name.c_str());
    }
}

int SnpeEngine::init(const std::string &model_path, int platdorm) {
    // 1.set runtime
    setruntime(platdorm);

    // 2. load model
    _container = zdl::DlContainer::IDlContainer::open(model_path);

    if (_container == nullptr) {
        std::cout << "load model error : " << zdl::DlSystem::getLastErrorString() << std::endl;
        return -1;
    }

    // zdl::DlSystem::TensorShapeMap inputShapeMap;
    // inputShapeMap.add("images", {1UL, static_cast<unsigned long>(input_image_size.width), static_cast<unsigned long>(input_image_size.height), 3UL});
    // 3. build engine
    zdl::SNPE::SNPEBuilder snpe_builder(_container.get());
    zdl::DlSystem::PerformanceProfile_t profile = zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;

    if (_output_tensor_names.size() != 0){
        snpe_builder.setOutputLayers(_output_tensor_names);
    }
    _engine = snpe_builder
            .setRuntimeProcessorOrder(_runtime_list)
            .setPerformanceProfile(profile)
            // .setCPUFallbackMode(true)
            .build();

    if (_engine == nullptr) {
        std::cout << "build engine error : " << zdl::DlSystem::getLastErrorString() << std::endl;
        return -1;
    }

    std::cout << "init success..." << std::endl;
    return 0;
}

void SnpeEngine::build_tensor(const cv::Mat &mat) {
    zdl::DlSystem::Dimension dims[4];
    dims[0] = 1;
    dims[1] = model_input_width;
    dims[2] = model_input_hight;
    dims[3] = 3;
    zdl::DlSystem::TensorShape tensorShape(dims, 4);
    _input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(tensorShape);

    // std::vector<float> input_data((float*)mat.datastart, (float*)mat.dataend);
    // std::copy(input_data.begin(), input_data.end(), _input_tensor->begin());

    size_t dataSize = (char*)mat.dataend - (char*)mat.datastart;
    std::copy_n(reinterpret_cast<const float*>(mat.datastart), dataSize / sizeof(float), _input_tensor->begin());

}

int SnpeEngine::inference()
{
    _output_tensor_map.clear();
    bool ret = _engine->execute(_input_tensor.get(), _output_tensor_map);
    if (!ret) {
        std::cerr << "engine inference error : " << zdl::DlSystem::getLastErrorString() << std::endl;
        return -1;
    }

    zdl::DlSystem::StringList tensorNames = _output_tensor_map.getTensorNames();
    if (tensorNames.size() == 0) {
        std::cerr << "No output tensors found" << std::endl;
        return -1;
    }

    // 遍历所有输出张量的名称
    for (const auto& tensorName : tensorNames) {
        // 从 _output_tensor_map 中获取对应的张量指针
        zdl::DlSystem::ITensor* tensor = _output_tensor_map.getTensor(tensorName);
        if (tensor != nullptr) {
            // 将张量指针存储到map中
            auto *pdata = reinterpret_cast<float *>(&(*tensor->begin()));
            zdl::DlSystem::TensorShape shape = tensor->getShape();
            _out_data_ptr[tensorName] = pdata;
            _output_shapes[tensorName] = shape;
        } else {
            std::cerr << "Tensor not found: " << tensorName << std::endl;
        }
    }
    return 0;
}

#if 0
int SnpeEngine::inference(const cv::Mat &cv_mat, std::vector<Yolov8OutPut>& outPut) {
    cv::Mat input_mat(cv_mat);
    float det_scale;
    double start = static_cast<double>(cv::getTickCount());
    //std::cout << "---------inference Preprocessing" << std::endl;
    Preprocessing(input_mat, det_scale);
    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "---------Preprocess Time cost : " << time_cost << "ms" << std::endl;

    start = static_cast<double>(cv::getTickCount());
    build_tensor(input_mat);
    end = static_cast<double>(cv::getTickCount());
    time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "---------Build Tensor Time cost : " << time_cost << "ms" << std::endl;

    start = static_cast<double>(cv::getTickCount());
    bool ret = _engine->execute(_input_tensor.get(), _output_tensor_map);
    if (!ret) {
        std::cerr << "engine inference error : " << zdl::DlSystem::getLastErrorString() << std::endl;
        return -1;
    }
    end = static_cast<double>(cv::getTickCount());
    time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "---------Inference Time cost : " << time_cost << "ms" << std::endl;

    start = static_cast<double>(cv::getTickCount());
    // 获取输出张量名称列表
    zdl::DlSystem::StringList tensorNames = _output_tensor_map.getTensorNames();
    if (tensorNames.size() == 0) {
        std::cerr << "No output tensors found" << std::endl;
        return -1;
    }

    const auto& boxes_tensorName = tensorNames.at(0);
    const auto& scores_tensorName = tensorNames.at(1);
    // std::cout << "tensorName0 : " << boxes_tensorName << std::endl;
    // std::cout << "tensorName1 : " << scores_tensorName << std::endl;

    zdl::DlSystem::ITensor *boxes_outTensor = _output_tensor_map.getTensor(boxes_tensorName);
    zdl::DlSystem::ITensor *scores_outTensor = _output_tensor_map.getTensor(scores_tensorName);
    if (!boxes_outTensor || !scores_outTensor) {
        std::cerr << "Failed to get output tensor" << std::endl;
        return -1;
    }

    if(3 == this->plantform || 2 == this->plantform) {
        Postprocessing(boxes_outTensor, scores_outTensor, outPut, det_scale);
    }
    else {
        Postprocessing(boxes_outTensor, outPut, det_scale);
    }
    _output_tensor_map.clear();

    end = static_cast<double>(cv::getTickCount());
    time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "---------Postprocess Time cost : " << time_cost << "ms" << std::endl;
    return 0;
}

void SnpeEngine::Postprocessing(zdl::DlSystem::ITensor *boxes_outTensor, zdl::DlSystem::ITensor *scores_outTensor, std::vector<Yolov8OutPut>& outPut, float det_scale) const{
    auto *boxes_data = reinterpret_cast<float *>(&(*boxes_outTensor->begin()));
    auto *scores_data = reinterpret_cast<float *>(&(*scores_outTensor->begin()));

    zdl::DlSystem::TensorShape boess_shape = boxes_outTensor->getShape();
    zdl::DlSystem::TensorShape scores_shape = scores_outTensor->getShape();
    if (boess_shape.rank() < 2 || scores_shape.rank() < 2) {
        std::cerr << "Invalid output tensor shape" << std::endl;
    }

    // std::cout << "tensor_dim: " << boess_shape[2] << " " << boess_shape[1] << std::endl;
    // std::cout << "tensor_dim: " << scores_shape[2] << " " << scores_shape[1] << std::endl;
    const cv::Mat boxes_mat = cv::Mat(cv::Size(static_cast<int>(boess_shape[2]), static_cast<int>(boess_shape[1])), CV_32F, boxes_data).t();
    cv::Mat scores_mat = cv::Mat(cv::Size(static_cast<int>(scores_shape[2]), static_cast<int>(scores_shape[1])), CV_32F, scores_data).t();

// sigmoid 使用循环耗时高于在cv图像上直接操作，使用cvSigmoid
#if 1
    cvSigmoid(scores_mat);
#else
    for (int i = 0; i < scores_mat.rows; ++i) {
        for (int j = 0; j < scores_mat.cols; ++j) {
            scores_mat.at<float>(i, j) = sigmoid(scores_mat.at<float>(i, j));
        }
    }
#endif

    auto boxes_pdata = reinterpret_cast<float*>(boxes_mat.data);
    auto scores_pdata = reinterpret_cast<float*>(scores_mat.data);

    const int rows = boxes_mat.rows;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels;
    const int socre_array_length = out_nums - 4;
    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, socre_array_length, CV_32FC1, scores_pdata);

        float box_x = boxes_pdata[0] / det_scale; //x
        float box_y = boxes_pdata[1] / det_scale; //y
        float box_w = boxes_pdata[2] / det_scale; //w
        float box_h = boxes_pdata[3] / det_scale; //h

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

    outPut.clear();
    for (std::vector<int>::size_type idx = 0; idx < nms_result.size(); idx++) {
        Yolov8OutPut result;
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.label = labels[idx];
        outPut.push_back(result);
    }

}

void SnpeEngine::Postprocessing(zdl::DlSystem::ITensor *outTensor, std::vector<Yolov8OutPut>& outPut, float det_scale) const{
    auto *data = reinterpret_cast<float *>(&(*outTensor->begin()));
    zdl::DlSystem::TensorShape tensor_shape = outTensor->getShape();
    if (tensor_shape.rank() < 2) {
        std::cerr << "Invalid output tensor shape" << std::endl;
    }

    const cv::Mat tensor_mat = cv::Mat(cv::Size(static_cast<int>(tensor_shape[2]), static_cast<int>(tensor_shape[1])), CV_32F, data).t();

    auto pdata = (float*)tensor_mat.data;

    const int rows = tensor_mat.rows;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels;
    const int socre_array_length = out_nums - 4;
    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, socre_array_length, CV_32FC1, pdata+4);

        float box_x = pdata[0] / det_scale; //x
        float box_y = pdata[1] / det_scale; //y
        float box_w = pdata[2] / det_scale; //w
        float box_h = pdata[3] / det_scale; //h

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
    double start = static_cast<double>(cv::getTickCount());
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, target_conf_th, nms_th, nms_result);
    auto end = static_cast<double>(cv::getTickCount());
    auto time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "---------Build Tensor Time cost : " << time_cost << "ms" << std::endl;

    outPut.clear();
    for (auto idx : nms_result) {
        Yolov8OutPut result;
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.label = labels[idx];
        outPut.push_back(result);
    }
}

void SnpeEngine::Preprocessing(cv::Mat &img, float &detScale) const{
    detScale = 1.0f;

// 使用填充来resize，耗时高于直接resize，可以参考 https://zhuanlan.zhihu.com/p/362701716 resize方式与训练时resize方式有关
#if 0
    // img.convertTo(img, CV_32FC3);
    if (img.cols != input_image_size.height || img.rows != input_image_size.width) {
        cv::Mat det_im = cv::Mat::zeros(input_image_size, CV_32FC3);
        int new_height, new_width;
        const float im_ratio = static_cast<float>(img.rows) / static_cast<float>(img.cols);
        if (im_ratio > 1) {
            new_height = input_image_size.height;
            new_width = static_cast<int>(new_height / im_ratio); // NOLINT(*-narrowing-conversions)
        }
        else {
            new_width = input_image_size.width;
            new_height = static_cast<int>(new_width * im_ratio); // NOLINT(*-narrowing-conversions)
        }
        detScale = static_cast<float>(new_height) / static_cast<float>(img.rows);
        cv::resize(img, img, cv::Size(new_width, new_height));
        img.copyTo(det_im(cv::Rect(0, 0, new_width, new_height)));
        img = det_im;
    }
#else
    cv::resize(img, img, cv::Size(model_input_hight, model_input_width));
#endif

// #if 0
//     if (cv::ocl::haveOpenCL()) {
//         std::cout << "use  opencl" << std::endl;
//         cv::UMat u_img = img.getUMat(cv::ACCESS_READ);
//
//         // 使用OpenCL加速计算blob
//         cv::UMat u_blob;
//         cv::dnn::blobFromImage(u_img, u_blob, 1.0/255, cv::Size(), cv::Scalar(103, 117, 123), true, false, CV_32F);
//         // 调整维度顺序
//         std::vector<int> order = {0, 2, 3, 1};
//         cv::UMat u_blob_transposed;
//         cv::transposeND(u_blob, order, u_blob_transposed);
//         img = u_blob_transposed.getMat(cv::ACCESS_READ);
//     }
// #else
//     img = cv::dnn::blobFromImage(img, 1.0/255, cv::Size(), cv::Scalar(103, 117, 123), false, false, CV_32F);
// # endif
//
// // 调整维度顺序,pc测试自写循环速度快于opencv函数，板端测差不多，这里使用自写循环
// #if 0
//     std::vector<int> order = {0, 2, 3, 1};
//     cv::transposeND(img, order, img);
// #else
//     int N = img.size[0];
//     int C = img.size[1];
//     int H = img.size[2];
//     int W = img.size[3];
//     cv::Mat output(N, H * W * C, CV_32F); // 创建输出矩阵
//     float* outputData = (float*)output.data;
//     // 遍历每一个维度
//     for(int n = 0; n < N; ++n) {
//         for(int h = 0; h < H; ++h) {
//             for(int w = 0; w < W; ++w) {
//                 for(int c = 0; c < C; ++c) {
//                     int inputIndex = n*(C*H*W) + c*(H*W) + h*W + w;
//                     int outputIndex = n*(H*W*C) + h*(W*C) + w*C + c;
//                     outputData[outputIndex] = ((float*)img.data)[inputIndex];
//                 }
//             }
//         }
//     }
//     img = output.reshape(N, {N, H, W, C}); // 调整形状使之符合NHWC格式
// #endif

    img.convertTo(img, CV_32F, 1.0/255);
    cv::Scalar meanValues(103.0 / 255, 117.0 / 255, 123.0 / 255);
    img -= meanValues;
    // 步骤3: BGR转RGB (如果需要)
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

}
#endif




// template<typename T>
// void SnpeEngine::Postprocessing(std::vector<T> & _results, float det_scale) const {
//     }
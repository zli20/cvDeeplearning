#ifndef SNPE_SNPEENGINE_H
#define SNPE_SNPEENGINE_H

#include <vector>
#include <iostream>
#include <any>

#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPEBuilder.hpp"

#include "SNPE/SNPE.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

class PostProcessing;
class PreProcessing;

enum Platform {CPU=0, GPU, DSP, AIP  };


class SnpeEngine
{
public:
    explicit SnpeEngine(const cv::Size image_size):
        model_input_size(image_size), model_input_width(image_size.width), model_input_hight(image_size.height)
    {
        checkRuntime();
    };

    virtual ~SnpeEngine(){
        _engine.reset();
        std:: cout << "deinit success..." << std::endl;
    };

    int init(const std::string &model_path, int platdorm=CPU);

    int inference(const cv::Mat& cv_mat, std::map<std::string, float*> &out_tensor);

    // virtual void postProcessing(std::vector<std::any> & _results, float det_scale) = 0;
    // virtual void Preprocessing(cv::Mat &img, float & det_scale);

    void setOutName(const std::vector<std::string>& out_names);

    void build_tensor(const cv::Mat& mat);

    int inference();

    cv::Size model_input_size;
    int model_input_width;
    int model_input_hight;
    Platform platform = CPU;

    // snpe model
    std::unique_ptr<zdl::SNPE::SNPE> _engine;
    std::unique_ptr<zdl::DlContainer::IDlContainer> _container;

    // snpe input & output
    zdl::DlSystem::StringList _output_tensor_names;

    std::unique_ptr<zdl::DlSystem::ITensor> _input_tensor;
    zdl::DlSystem::TensorMap _output_tensor_map;
    std::map<std::string, float*> _out_data_ptr;
    std::map<std::string, zdl::DlSystem::TensorShape > _output_shapes;

    // snpe builder config
    zdl::DlSystem::RuntimeList _runtime_list;

    void setruntime(int platdorm);

    static  void checkRuntime();

private:

    // void Preprocessing(cv::Mat &img, float &detScale) const;
    //
    // void Postprocessing(zdl::DlSystem::ITensor *boxes_outTensor, zdl::DlSystem::ITensor *scores_outTensor, std::vector<Yolov8OutPut>& outPut, float det_scale) const;
    //
    // void Postprocessing(zdl::DlSystem::ITensor *outTensor, std::vector<Yolov8OutPut>& outPut, float det_scale) const;



};

#endif //SNPE_SNPEENGINE_H

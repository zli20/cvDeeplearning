#ifndef INFERENCEONNX_H
#define INFERENCEONNX_H

#include <string>
#include <vector>
#include <map>

#include <onnxruntime_cxx_api.h>
#include<opencv2/opencv.hpp>

class InferenceOnnx {
private:
    Ort::Env* _env;
    Ort::Session* _session;

    std::string _model_path;

    int _init(Ort::Env* env = nullptr, const std::string& model_path = "", GraphOptimizationLevel opt_level = ORT_ENABLE_ALL, int threads = 0);

public:
    int _input_count = 0;
    int _output_count = 0;

    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;

    // input dims
    std::map<std::string, std::vector<int64_t>> _input_shapes;
    // output dims
    std::map<std::string, std::vector<int64_t>> _output_shapes;

    InferenceOnnx();
    explicit InferenceOnnx(const char* model_path);
    InferenceOnnx(Ort::Env*& env, const std::string& model_path);
    ~InferenceOnnx();

    int init();
    int init(Ort::Env*& env);
    int init(const std::string& model_path, GraphOptimizationLevel opt_level = ORT_ENABLE_ALL, int threads = 0);
    int init(Ort::Env*& env, const std::string& model_path, GraphOptimizationLevel opt_level = ORT_ENABLE_ALL, int threads = 0);

    void release();

    std::string getModelPath()const;
    void setModelPath(const std::string & model_path);

    // void setOptions(GraphOptimizationLevel opt_level, int threads = 0);
    std::vector<Ort::Value> Inference(const cv::Mat& input)const;

    bool isModelInitialized()const;

    //virtual void inputPreProcessing(cv::Mat& image, size_t img_size = 640) = 0;
    //virtual void outputPostProcessing() = 0;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& v) {
    size_t count = 0;
    os << "[ ";
    for (auto item : v) {
        os << item;
        if (++count < v.size())
            os << ", ";
    }
    os << " ]";

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const InferenceOnnx& model) {
    if (!model._input_count)
        os << "ONNX Model presents no inputs, model is probably not initialized" << std::endl;
    else {
        os << "Loaded Model Parameters [" << std::endl
            << "\t Input Count: " << model._input_count << std::endl
            << "\t Output Count: " << model._output_count << std::endl
            << "\t Inputs and Shapes: [" << std::endl;

        for (const auto& item : model._input_shapes)
            os << "\t\t" << item.first << ": " << item.second << std::endl;

        os << "\t ]" << std::endl << "\t Outputs and Shapes: [";
        if (!model._output_shapes.empty()) os << std::endl;

        for (const auto& item : model._output_shapes)
            os << "\t\t" << item.first << ": " << item.second << std::endl;

        os << "\t ]" << std::endl;
        os << "]";
    }

    return os;
}

#endif
#include <cstdlib>
#include <InferenceOnnx.h>

InferenceOnnx::InferenceOnnx() : _env(nullptr), _session(nullptr){}

InferenceOnnx::InferenceOnnx(const char* model_path) : _env(nullptr), _session(nullptr), _model_path(model_path) { std::cout << "Model Path: " << this->_model_path << std::endl; }

InferenceOnnx::InferenceOnnx(Ort::Env*& env, const std::string& model_path) : _env(env), _session(nullptr), _model_path(model_path) {}

InferenceOnnx::~InferenceOnnx() {
    release();
}

static int getNameIndex(const std::string& name, const std::vector<std::string>& vec) {
    for (size_t i = 0; i < vec.size(); i++)
        if (name == vec[i]) return i;
    return -1;
}

int InferenceOnnx::_init(Ort::Env* env, const std::string& model_path, GraphOptimizationLevel opt_level, int threads) {

    Ort::AllocatorWithDefaultOptions allocator_a;
    Ort::SessionOptions options;
    OrtThreadingOptions* thOpts;
    OrtStatusPtr status;
    status = Ort::GetApi().CreateThreadingOptions(&thOpts);
    status = Ort::GetApi().SetGlobalIntraOpNumThreads(thOpts, threads);
    if (status == nullptr) {
        std::cout << status << std::endl;
    }

//    Ort::GetApi().SetGlobalIntraOpNumThreads(thOpts, threads);
//    Ort::GetApi().CreateThreadingOptions(&thOpts);

    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.DisablePerSessionThreads();
    options.SetGraphOptimizationLevel(opt_level);
    options.SetInterOpNumThreads(threads);
    options.SetIntraOpNumThreads(threads);

    std::cout << "Defining ONNX Session Options" << std::endl;

    if (env == nullptr) {
        if (this->_env == nullptr)
            this->_env = new Ort::Env(thOpts, ORT_LOGGING_LEVEL_FATAL, "_env");
    }
    else {
        this->_env = env;
    }

    std::cout << "Loading ONNX model" << std::endl;
    if (model_path.empty()) {
        if (!this->_model_path.empty()) {
            // wchar_t path[1024];
            // mbstowcs_s(nullptr, path, 1024, this->model_path.c_str(), 1024);
            this->_session = new Ort::Session(*(this->_env), this->_model_path.c_str(), options);
        }
        else {
            std::cout << "No model path provided" << std::endl;
            return EXIT_FAILURE;
        }
    }
    else {
        // wchar_t path[1024];
        // mbstowcs_s(nullptr, path, 1024, model_path.c_str(), 1024);
        this->_session = new Ort::Session(*(this->_env), model_path.c_str(), options);
    }

    std::cout << "Getting Model Input/Output Info" << std::endl;
    this->_input_count = this->_session->GetInputCount();
    this->_output_count = this->_session->GetOutputCount();


    for (int i = 0; i < this->_input_count; i++) {
        char* input = this->_session->GetInputName(i, allocator_a);
        this->_input_names.emplace_back(input);
        allocator_a.Free((void*)input);
    }

    for (int i = 0; i < this->_output_count; i++) {
        char* output = this->_session->GetOutputName(i, allocator_a);
        this->_output_names.emplace_back(output);
        allocator_a.Free((void*)output);
    }

    if (static_cast<int>(this->_input_names.size()) != this->_input_count || static_cast<int>(this->_output_names.size()) != this->_output_count)
        return EXIT_FAILURE;


    for (const auto& name : this->_input_names) {
        int idx = getNameIndex(name, this->_input_names);
        if (idx < 0) continue;

        Ort::TypeInfo inputTypeInfo = this->_session->GetInputTypeInfo(idx);
        auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();

        if (shape.empty()) {
            std::cerr << "Error: The shape vector for input index " << idx << " is empty." << std::endl;
            return EXIT_FAILURE;
        }
        bool hasDynamicDimension = std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return (dim == -1 || dim == 0); });
        if (!hasDynamicDimension) {
            this->_input_shapes.emplace(name, shape);
        }
        else {
            std::cerr << "Error: Dynamic dimensions are not supported for input '" << name << "'." << std::endl;
            return EXIT_FAILURE;
        }
    }

    for (const auto& name : this->_output_names) {
        int idx = getNameIndex(name, this->_output_names);
        if (idx < 0) continue;
        this->_output_shapes.emplace(name, this->_session->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape());
    }

    std::cout << "Releasing Threading Options" << std::endl;
    Ort::GetApi().ReleaseThreadingOptions(thOpts);

    return EXIT_SUCCESS;
}

int InferenceOnnx::init() {
    return _init();
}

int InferenceOnnx::init(Ort::Env*& env) {
    return _init(env);
}

int InferenceOnnx::init(const std::string& model_path, GraphOptimizationLevel opt_level, int threads) {
    return _init(this->_env, model_path, opt_level, threads);
}

int InferenceOnnx::init(Ort::Env*& env, const std::string& model_path, GraphOptimizationLevel opt_level, int threads) {
    return _init(env, model_path, opt_level, threads);
}

void InferenceOnnx::release() {
    delete this->_env;
    delete this->_session;
}

std::string InferenceOnnx::getModelPath()const {
    return this->_model_path;
}

void InferenceOnnx::setModelPath(const std::string & model_path) {
    this->_model_path = model_path;
}

std::vector<Ort::Value> InferenceOnnx::Inference(const cv::Mat& input)const{
    if (!this->_env || !this->_session)
        throw std::runtime_error("Environment or session is not initialized.");

    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> dims(this->_input_shapes.at(this->_input_names[0]));
    size_t data_size = 1;
    for (auto dim : dims) data_size *= dim;

    // same as reinterpreted_cast<const float*>
    std::vector<float> input_data(data_size);
    input_data.assign((float*)input.datastart, (float*)input.dataend);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), dims.data(), dims.size());
    if (!input_tensor.IsTensor()) {
        throw std::runtime_error("Error: Build Tensor Failed!");
    }

    std::vector<const char*> input_nm(this->_input_count);
    for (int i = 0; i < this->_input_count; i++)
        input_nm[i] = this->_session->GetInputName(i, allocator);

    std::vector<const char*> output_names(this->_output_count);
    for (int i = 0; i < this->_output_count; i++)
        output_names[i] = this->_session->GetOutputName(i, allocator);
    // for (auto name : output_names) std::cout << name << std::endl;

    auto output_tensor = this->_session->Run(Ort::RunOptions{ nullptr }, input_nm.data(), &input_tensor, 1, output_names.data(), output_names.size());
    assert(output_tensor.front().IsTensor());

    //ADD THE ALLOCATOR.FREE to FREE THE INPUT AND OUTPUT NAMES!
    for (auto& name : input_nm) allocator.Free((void*)name);
    for (auto& name : output_names) allocator.Free((void*)name);

    return output_tensor;
}

bool InferenceOnnx::isModelInitialized()const {
    return(this->_session != nullptr);
}

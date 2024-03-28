#include <cstdlib>
#include <InferenceOnnx.h>

InferenceOnnx::InferenceOnnx() : env(nullptr), session(nullptr){}

InferenceOnnx::InferenceOnnx(const char* model_path) : model_path(model_path), env(nullptr), session(nullptr) { std::cout << "Model Path: " << this->model_path << std::endl; }

InferenceOnnx::InferenceOnnx(Ort::Env*& env, const std::string& model_path) : env(env), model_path(model_path), session(nullptr) {}

InferenceOnnx::~InferenceOnnx() {
    release();
}

static int getNameIndex(const std::string& name, const std::vector<std::string>& vec) {
    for (int i = 0; i < vec.size(); i++)
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
    // status = Ort::GetApi().SetGlobalInterOpNumThreads(thOpts, threads);

    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.DisablePerSessionThreads();
    options.SetGraphOptimizationLevel(opt_level);
    options.SetInterOpNumThreads(threads);
    options.SetIntraOpNumThreads(threads);

    std::cout << "Defining ONNX Session Options" << std::endl;

    if (env == nullptr) {
        if (this->env == nullptr)
            this->env = new Ort::Env(thOpts, ORT_LOGGING_LEVEL_FATAL, "retina_env");
    }
    else {
        this->env = env;
    }

    std::cout << "Loading ONNX model" << std::endl;
    if (model_path.empty()) {
        if (!this->model_path.empty()) {
            // wchar_t path[1024];
            // mbstowcs_s(nullptr, path, 1024, this->model_path.c_str(), 1024);
            this->session = new Ort::Session(*(this->env), this->model_path.c_str(), options);
        }
        else {
            std::cout << "No model path provided" << std::endl;
            return EXIT_FAILURE;
        }
    }
    else {
        // wchar_t path[1024];
        // mbstowcs_s(nullptr, path, 1024, model_path.c_str(), 1024);
        this->session = new Ort::Session(*(this->env), model_path.c_str(), options);
    }

    std::cout << "Getting Model Input/Output Info" << std::endl;
    this->input_count = this->session->GetInputCount();
    this->output_count = this->session->GetOutputCount();


    for (int i = 0; i < this->input_count; i++) {
        char* input = this->session->GetInputName(i, allocator_a);
        this->input_names.push_back(input);
        allocator_a.Free((void*)input);
    }

    for (int i = 0; i < this->output_count; i++) {
        char* output = this->session->GetOutputName(i, allocator_a);
        this->output_names.push_back(output);
        allocator_a.Free((void*)output);
    }

    if (this->input_names.size() != this->input_count || this->output_names.size() != this->output_count) return EXIT_FAILURE;


    for (const auto& name : this->input_names) {
        int idx = getNameIndex(name, this->input_names);
        if (idx < 0) continue;

        Ort::TypeInfo inputTypeInfo = this->session->GetInputTypeInfo(idx);
        auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();

        if (shape.empty()) {
            std::cerr << "Error: The shape vector for input index " << idx << " is empty." << std::endl;
            return EXIT_FAILURE;
        }
        bool hasDynamicDimension = std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return (dim == -1 || dim == 0); });
        if (!hasDynamicDimension) {
            this->input_shapes.emplace(name, shape);
        }
        else {
            std::cerr << "Error: Dynamic dimensions are not supported for input '" << name << "'." << std::endl;
            return EXIT_FAILURE;
        }
    }

    for (const auto& name : this->output_names) {
        int idx = getNameIndex(name, this->output_names);
        if (idx < 0) continue;
        this->output_shapes.emplace(name, this->session->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape());
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
    return _init(this->env, model_path, opt_level, threads);
}

int InferenceOnnx::init(Ort::Env*& env, const std::string& model_path, GraphOptimizationLevel opt_level, int threads) {
    return _init(env, model_path, opt_level, threads);
}

void InferenceOnnx::release() {
    delete this->env;
    delete this->session;
}

std::string InferenceOnnx::getModelPath()const {
    return this->model_path; 
}

void InferenceOnnx::setModelPath(const std::string & model_path) {
    this->model_path = model_path;
}

std::vector<Ort::Value> InferenceOnnx::Inference(const cv::Mat& input)const{
    if (!env || !session)
        throw std::runtime_error("Environment or session is not initialized.");

    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> dims(this->input_shapes.at(this->input_names[0]));
    size_t data_size = 1;
    for (auto dim : dims) data_size *= dim;

    // same as reinterpreted_cast<const float*>
    std::vector<float> input_data(data_size);
    input_data.assign((float*)input.datastart, (float*)input.dataend);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), dims.data(), dims.size());
    if (!input_tensor.IsTensor()) {
        throw std::runtime_error("Error: Build Tensor Failed!");
    }

    std::vector<const char*> input_nm(this->input_count);
    for (int i = 0; i < this->input_count; i++)
        input_nm[i] = this->session->GetInputName(i, allocator);

    std::vector<const char*> output_names(this->output_count);
    for (int i = 0; i < this->output_count; i++) 
        output_names[i] = this->session->GetOutputName(i, allocator);
    // for (auto name : output_names) std::cout << name << std::endl;

    auto output_tensor = this->session->Run(Ort::RunOptions{ nullptr }, input_nm.data(), &input_tensor, 1, output_names.data(), output_names.size());
    assert(output_tensor.front().IsTensor());

    //ADD THE ALLOCATOR.FREE to FREE THE INPUT AND OUTPUT NAMES!
    for (auto& name : input_nm) allocator.Free((void*)name);
    for (auto& name : output_names) allocator.Free((void*)name);

    return output_tensor;
}

bool InferenceOnnx::isModelInitialized()const {
    return(this->session != nullptr);
}

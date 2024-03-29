/*
int main(int argc, char* argv[]) {
    // 检查是否有足够的命令行参数
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " int <platform> " << " string <modelpath> " << " string <inputpath> " << " int <mode>" << std::endl;
        std::cerr << "platform: 0->CPU 1->GPU 2->DSP 3->AIP" << std::endl;
        std::cerr << "mode: 0->Face Detection 1->Object Detection" << std::endl;
        return 1;
    }

    // 从命令行参数中获取输入
    int platform = std::stoi(argv[1]);
    std::string modelpath = argv[2];
    std::string img_path = argv[3];
    int mode = std::stoi(argv[4]);

    std::cout << "platform: " << platform << std::endl;
    std::cout << "modelpath: " << modelpath << std::endl;
    std::cout << "inputpath: " << img_path << std::endl;

    // 创建相应的引擎对象
    EngineBase* engine = nullptr;
    if (mode == 0) {
        engine = new Yolov8FaceSnpe(modelpath, cv::Size(640, 640), platform);
    } else if (mode == 1) {
        engine = new Yolov8DetSnpe(modelpath, cv::Size(640, 640), platform);
    } else {
        std::cerr << "Invalid mode. Please specify either 0 (Face Detection) or 1 (Object Detection)." << std::endl;
        return 1;
    }

    std::vector<RESULT> results;

    // 处理图像或视频
    // ...

    // 清理资源
    delete engine;
    engine = nullptr;

    return 0;
}
*/

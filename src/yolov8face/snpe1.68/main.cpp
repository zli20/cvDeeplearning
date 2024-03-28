#include<iostream>
#include <string>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "Yolov8FaceSnpe.h"
#include "Datatype.h"

int main(int argc, char* argv[]) {
    // 检查是否有足够的命令行参数
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <platdorm> " << " <modelpath> " << " <inputpath> "<< std::endl;
        return 1;
    }

    std::string modelpath;
    modelpath = argv[2];
    std::cout << "modelpath: " << modelpath << std::endl;

    std::string img_path;
    img_path = argv[3];
    std::cout << "inputpath: " << img_path << std::endl;
    return 0;

}
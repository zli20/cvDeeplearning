#include<iostream>
#include <string>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "RetinafaceSnpe.h"
#include "Datatype.h"


int platform;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " string <cfg path> " << " string <input path> " << std::endl;
        return -1;
    }
    std::string cfg_path;
    cfg_path = argv[1];
    std::cout << "cfg path: " << cfg_path << std::endl;

    Maincfg& cfg = Maincfg::instance();
    cfg.LoadCfg(cfg_path);

    std::string img_path;
    img_path = argv[2];
    std::cout << "inputpath: " << img_path << std::endl;
    auto *_engine = new RetinafaceSnpe();

    std::vector<FACE_RESULT> results;

    size_t dotPos = img_path.find_last_of('.');
    if (dotPos == std::string::npos) {
        std::cout << "Error: File has no extension." << std::endl;
        return 0;
    }
    std::string extension = img_path.substr(dotPos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (extension == "jpg" || extension == "png" || extension == "jpeg") {
        std::cout << "Processing image: " << img_path << std::endl;
        cv::Mat cvmat = cv::imread(img_path);

        // To test the inference speed
        int nums = 1;
        while(nums > 0) {
            cv::Mat result_mat = cvmat.clone();
            auto start = static_cast<double>(cv::getTickCount());
            _engine->getInference(result_mat, results);
            auto end = static_cast<double>(cv::getTickCount());
            double time_cost = (end - start) / cv::getTickFrequency() * 1000;
            std::cout << "--------------------------All Time cost : " << time_cost << "ms" << std::endl;
            nums --;
        }
        _engine->drawResult(cvmat, results);

        // cv::imwrite("../images/result_mat.jpg", result_mat);

        // imshow("yolov8", cvmat);
        // cv::waitKey(0);

        cv::Mat resized_img;
        if (cvmat.cols != 640 || cvmat.rows != 640) {
            int new_height, new_width;
            float im_ratio = float(cvmat.rows) / float(cvmat.cols);
            if (im_ratio > 1) {
                new_height = 640;
                new_width = int(new_height / im_ratio);
            }
            else {
                new_width = 640;
                new_height = int(new_width * im_ratio);
            }

            cv::resize(cvmat, resized_img, cv::Size(new_width, new_height));
        }
        imshow("yolov8n", resized_img);
        cv::waitKey(0);
    }
    else if (extension == "avi" || extension == "mp4") {
        std::cout << "Processing video: " << img_path << std::endl;
        cv::VideoCapture capture;
        capture.open(img_path);
        if (!capture.isOpened()) {
            printf("could not read this video file...\n");
            return -1;
        }

        int num_frames = 0;
        cv::VideoWriter video("out0.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(1920,1080));

        while (true) {
            cv::Mat frame;
            if (!capture.read(frame))
            {
                if(0 == num_frames){std::cout<<"\n Cannot read the video file. please check your video.\n";}
                else{std::cout<<"\n End read video.\n";}
                break;
            }
            cv::Mat result_mat = frame.clone();
            num_frames ++;
            // if(num_frames > 100) {
            //     break;
            // }
            auto start = static_cast<double>(cv::getTickCount());
            _engine->getInference(frame, results);
            auto end = static_cast<double>(cv::getTickCount());
            double time_cost = (end - start) / cv::getTickFrequency() * 1000;
            std::cout << "---------Inference Time cost : " << time_cost << "ms" << std::endl;

            _engine->drawResult(result_mat, results);

            cv::imshow("YOLOv8: ", result_mat);
            if(cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
            {
                break;
            }

            // video.write(result_mat);
            results.clear();
        }
        capture.release();
        video.release();
    } else {std::cout << "Unsupported file format: " << img_path << std::endl;}


    delete _engine;
    _engine = nullptr;
    // cv::destroyAllWindows();
    return 0;
}


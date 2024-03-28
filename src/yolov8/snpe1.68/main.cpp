
#include <iostream>
#include <string>
#include <fstream>

#include "Yolov8DetSnpe.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "Datatype.h"

int platform;
//
// void test_bytetrack(cv::Mat& frame, std::vector<Yolov8OutPut>& results,BYTETracker& tracker){
//     std::vector<Yolov8OutPut> objects;
//
//     for (Yolov8OutPut dr : results)
//     {
//
//         if(dr.label == 2)
//         {
//             objects.push_back(dr);
//         }
//     }
//
//     std::vector<STrack> output_stracks = tracker.update(objects);
//
//     for (unsigned long i = 0; i < output_stracks.size(); i++)
//     {
//         std::vector<float> tlwh = output_stracks[i].tlwh;
//         bool vertical = tlwh[2] / tlwh[3] > 1.6;
//         if (tlwh[2] * tlwh[3] > 20 && !vertical)
//         {
//             cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
//             cv::putText(frame, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
//                     0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
//             cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
//         }
//     }
// }
//
// void test_deepsort(cv::Mat& frame, std::vector<Yolov8OutPut>& results,DeepsortTracker& mytracker)
// {
//     std::vector<Yolov8OutPut> objects;
//
//     DETECTIONS detections;
//     for (Yolov8OutPut dr : results)
//     {
//         //cv::putText(frame, classes[dr.classId], cv::Point(dr.box.tl().x+10, dr.box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(0, 255, 0));
//         if(dr.label == 2) //person
//         {
//             objects.push_back(dr);
//             // cv::rectangle(frame, dr.box, cv::Scalar(255, 0, 0), 2);
//             DETECTION_ROW tmpRow;
//             tmpRow.tlwh = DETECTBOX(dr.box.x, dr.box.y,dr.box.width,  dr.box.height);
//             tmpRow.confidence = dr.confidence;
//             detections.push_back(tmpRow);
//         }
//     }
//
//     std::cout<<"begin track"<<std::endl;
//     std::string feature_model_path = "../models/DSort_168_128_64.dlc";
//     if(FeatureTensor::getInstance(feature_model_path)->getRectsFeature(frame, detections))
//     {
//         std::cout << "get feature succeed!"<<std::endl;
//         mytracker.predict();
//         mytracker.update(detections);
//         std::vector<RESULT_DATA> result;
//         for(DTrack& track : mytracker.tracks) {
//             // if(!track.is_confirmed() || track.time_since_update > 1) continue;
//             result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
//         }
//         for(unsigned int k = 0; k < detections.size(); k++)
//         {
//             // DETECTBOX tmpbox = detections[k].tlwh;
//             // cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
//             // cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
//             // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B
//
//             for(unsigned int k = 0; k < result.size(); k++)
//             {
//                 DETECTBOX tmp = result[k].second;
//                 cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
//                 rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);
//
//                 std::string label = cv::format("%d", result[k].first);
//                 cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
//             }
//         }
//     }
//     std::cout<<"end track"<<std::endl;
// }

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
    auto *YoloV8_engine = new Yolov8DetSnpe(modelpath, cv::Size(640, 640), platform);

    std::vector<DET_RESULT> results;

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

        int nums = 1;
        while(nums > 0) {
            cv::Mat result_mat = cvmat.clone();
            auto start = static_cast<double>(cv::getTickCount());
            YoloV8_engine->getInference(result_mat, results);
            auto end = static_cast<double>(cv::getTickCount());
            double time_cost = (end - start) / cv::getTickFrequency() * 1000;
            std::cout << "--------------------------All Time cost : " << time_cost << "ms" << std::endl;
            nums --;
        }
        YoloV8_engine->drawResult(cvmat, results);

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
        // 显示调整后的图像
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
            YoloV8_engine->getInference(frame, results);
            auto end = static_cast<double>(cv::getTickCount());
            double time_cost = (end - start) / cv::getTickFrequency() * 1000;
            std::cout << "---------Inference Time cost : " << time_cost << "ms" << std::endl;

            YoloV8_engine->drawResult(result_mat, results);

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


    delete YoloV8_engine;
    YoloV8_engine = nullptr;
    // cv::destroyAllWindows();
    return 0;
}


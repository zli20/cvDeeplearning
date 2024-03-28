#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include<fstream>
#include<opencv2/opencv.hpp>

// coco 80 class namees
extern const std::vector<std::string> cocoClassNamesList;

// coco pose 17 keypoints line and colors
//          1       2
//      3       0       4
//           
//          5       6
//     
//      7               8 
//     
//    9                  10 
//         11      12
//     
//         13      14  
//    
//         15      16
extern const std::vector<std::vector<unsigned int>> SKELETON;
extern const std::vector<std::vector<unsigned int>> KPS_COLORS;
extern const std::vector<std::vector<unsigned int>> LIMB_COLORS;

cv::Mat hwc_to_chw(const cv::Mat& image);

void chw_to_hwc(cv::InputArray src, cv::OutputArray dst);

// resize image with padding 
void resize_padding(cv::Mat& img, float& det_scale, cv::Size img_size);

#endif

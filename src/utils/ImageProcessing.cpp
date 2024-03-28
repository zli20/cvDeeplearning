#include <ImageProcessing.h>

const std::vector<std::string> cocoClassNamesList = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

const std::vector<std::vector<unsigned int>> KPS_COLORS = {
{0,   255, 0},
 {0,   255, 0},
 {0,   255, 0},
 {0,   255, 0},
 {0,   255, 0},
 {255, 128, 0},
 {255, 128, 0},
 {255, 128, 0},
 {255, 128, 0},
 {255, 128, 0},
 {255, 128, 0},
 {51,  153, 255},
 {51,  153, 255},
 {51,  153, 255},
 {51,  153, 255},
 {51,  153, 255},
 {51,  153, 255} };

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6,  12},
                                                         {7,  13},
                                                         {6,  7},
                                                         {6,  8},
                                                         {7,  9},
                                                         {8,  10},
                                                         {9,  11},
                                                         {2,  3},
                                                         {1,  2},
                                                         {1,  3},
                                                         {2,  4},
                                                         {3,  5},
                                                         {5,  7},
                                                         {4,  6}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = { {51,  153, 255},
                                                            {51,  153, 255},
                                                            {51,  153, 255},
                                                            {51,  153, 255},
                                                            {255, 51,  255},
                                                            {255, 51,  255},
                                                            {255, 51,  255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0} };
   
cv::Mat hwc_to_chw(const cv::Mat &image){

    cv::Mat chw_image(image.size(), image.type());

    for(int i = 0; i < image.channels(); i++)
        cv::extractChannel(
            image,
            cv::Mat(
                image.size().height,
                image.size().width,
                CV_32FC1,
                &(chw_image.at<float>(image.size().height*image.size().width*i))
            ),
            i
        );

    return chw_image;
}

void chw_to_hwc(cv::InputArray src, cv::OutputArray dst){
     const auto& src_size = src.getMat().size;
     const int src_c = src_size[0];
     const int src_h = src_size[1];
     const int src_w = src_size[2];

     auto c_hw = src.getMat().reshape(0, {src_c, src_h * src_w});

     dst.create(src_h, src_w, CV_MAKE_TYPE(src.depth(), src_c));
     cv::Mat dst_1d = dst.getMat().reshape(src_c, {src_h, src_w});

     cv::transpose(c_hw, dst_1d);
 }
 
void resize_padding(cv::Mat &img, float& det_scale, cv::Size img_size) {
     cv::Mat det_im = cv::Mat::zeros(img_size, CV_32FC3);
     cv::Mat resized_img;
     int new_height, new_width;
     float im_ratio = float(img.rows) / float(img.cols);
     if (im_ratio > 1) {
         new_height = img_size.height;
         new_width = int(new_height / im_ratio);
     }
     else {
         new_width = img_size.width;
         new_height = int(new_width * im_ratio);
     }
     det_scale = (float)(new_height) / (float)(img.rows);
     cv::resize(img, resized_img, cv::Size(new_width, new_height));
     resized_img.copyTo(det_im(cv::Rect(0, 0, new_width, new_height)));
     img = det_im;
 }




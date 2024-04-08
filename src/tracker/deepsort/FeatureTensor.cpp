#include "FeatureTensor.h"
#include <iostream>

extern int platform;

FeatureTensor *FeatureTensor::instance = nullptr;

FeatureTensor *FeatureTensor::getInstance(const std::string& model_path)
{
    if (instance == nullptr)
    {
        instance = new FeatureTensor(model_path);
    }
    return instance;
}
FeatureTensor::FeatureTensor(const std::string& model_path)
{
    // prepare model:
    bool status = init(model_path);
    if (status == false)
    {
        std::cout << "init failed" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "init succeed" << std::endl;
    }
}
FeatureTensor::~FeatureTensor()
= default;


bool FeatureTensor::init(const std::string& model_path)
{
    feature_net = std::make_unique<SnpeEngine>(width_, height_);
    std::cout << "FeatureTensor::init() " << std::endl;
    feature_net->init(model_path, platform);
    return true;
}

// void FeatureTensor::preprocess(cv::Mat &imageBGR, std::vector<float> &inputTensorValues, size_t &inputTensorSize)
void FeatureTensor::preprocess(cv::Mat &imageBGR)
{

    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;

    cv::resize(imageBGR, resizedImageBGR,
                cv::Size(width_, height_),
                cv::InterpolationFlags::INTER_CUBIC);

    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);

    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    // step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);
    feature_net->build_tensor(resizedImage);
}

bool FeatureTensor::getRectsFeature(const cv::Mat &img, DETECTIONS &d)
{
    for (DETECTION_ROW &dbox : d)
    {
        cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                               int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
        rc.width = rc.height * 0.5;
        rc.x = (rc.x >= 0 ? rc.x : 0);
        rc.y = (rc.y >= 0 ? rc.y : 0);
        rc.width = (rc.x + rc.width <= img.cols ? rc.width : (img.cols - rc.x));
        rc.height = (rc.y + rc.height <= img.rows ? rc.height : (img.rows - rc.y));

        cv::Mat mattmp = img(rc).clone();
        preprocess(mattmp);
        this->feature_net->inference();
        auto *data = this->feature_net->_out_data_ptr["207"];
        for (int i = 0; i < k_feature_dim; i++)
        {
            // dbox.feature[i] = f[i];
            dbox.feature[i] = data[i];
        }
    }

    return true;
}

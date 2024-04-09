
/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 02:39:47
*/
#include "trackDataType.h"
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept>
#include "opencv2/opencv.hpp"
#include <opencv2/dnn/dnn.hpp>
#include "SnpeEngine.h"

typedef unsigned char uint8;

template <typename T>
T vectorProduct(const std::vector<T> &v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

class FeatureTensor
{
public:
    static FeatureTensor *getInstance();
    bool getRectsFeature(const cv::Mat &img, DETECTIONS &d);
    // void preprocess(cv::Mat &imageBGR, std::vector<float> &inputTensorValues, size_t &inputTensorSize);
    bool init(const std::string& model_path, int platform, int height_=64, int width_=128);
private:
    FeatureTensor();
    FeatureTensor(const FeatureTensor &);
    FeatureTensor &operator=(const FeatureTensor &);
    static FeatureTensor *instance;
    ~FeatureTensor();

    // void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

    void preprocess(cv::Mat &imageBGR) const;
public:
    // void test();

    int height_ = 64;
    int width_ = 128;
    std::unique_ptr<SnpeEngine> feature_net;

};

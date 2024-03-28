
/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 02:39:47
*/
#include "dataType.h"
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
    static FeatureTensor *getInstance(const std::string& model_path);
    bool getRectsFeature(const cv::Mat &img, DETECTIONS &d);
    // void preprocess(cv::Mat &imageBGR, std::vector<float> &inputTensorValues, size_t &inputTensorSize);


private:
    explicit FeatureTensor(const std::string& model_path);
    FeatureTensor(const FeatureTensor &);
    FeatureTensor &operator=(const FeatureTensor &);
    static FeatureTensor *instance;
    bool init(const std::string& model_path);
    ~FeatureTensor();

    // void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

    void preprocess(cv::Mat &imageBGR);
public:
    // void test();

    static constexpr int height_ = 64;
    static constexpr int width_ = 128;

    std::unique_ptr<SnpeEngine> feature_net;
};

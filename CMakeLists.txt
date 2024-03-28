cmake_minimum_required(VERSION 3.10)
project(cvDL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -DNDEBUG -fopenmp")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

set (OPENCV_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/../third_party/Opencv4.6/include")
set (SNPE_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/../third_party/snpe1.68/include/zdl")
set (ONNXRUNTIME_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/../third_party/onnx/include")
set (EIGEN_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/../third_party/eigen3")

set (OPENCV_LIB_PATH "${CMAKE_SOURCE_DIR}/../third_party/Opencv4.6/lib")
set (SNPE_LIB_PATH "${CMAKE_SOURCE_DIR}/../third_party/snpe1.68/lib/x86_64-linux-clang")
set (ONNXRUNTIME_LIB_PATH "${CMAKE_SOURCE_DIR}/../third_party/onnx/lib")

endif ()

set(ONNXRUNTIME_LIBS libonnxruntime.so)
set(SNPE_LIBS libSNPE.so)
#set(OPENCV_LIBS opencv_core opencv_dnn opencv_highgui opencv_imgcodecs opencv_imgproc)
set(OPENCV_LIBS opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann opencv_gapi opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml
        opencv_objdetect opencv_photo opencv_stitching opencv_videoio opencv_video)

add_subdirectory(src)







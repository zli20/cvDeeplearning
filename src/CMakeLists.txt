cmake_minimum_required(VERSION 3.10)


FILE(GLOB UTILS_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h")
FILE(GLOB UTILS_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp")
set(UTILS_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/utils/")

#FILE(GLOB ONNXRUNTIME_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/snpe1.68/*.h")
#FILE(GLOB SNPE1.68_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/onnxruntime/*.h")
#FILE(GLOB ONNXRUNTIME_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/snpe1.68/*.cpp")
#FILE(GLOB SNPE1.68_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/onnxruntime/*.cpp")

message(STATUS "OpenCV_INCLUDE_DIRS = ${OPENCV_INCLUDE_PATH}")

#add_subdirectory(midas)
#add_subdirectory(pfld)
#add_subdirectory(retinaface)
add_subdirectory(yolov8)
#add_subdirectory(yolov8face)
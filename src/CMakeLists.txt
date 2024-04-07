cmake_minimum_required(VERSION 3.10)


FILE(GLOB UTILS_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h")
FILE(GLOB UTILS_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp")
set(UTILS_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/utils/")

include_directories(
        ${OPENCV_INCLUDE_PATH}
        ${ONNXRUNTIME_INCLUDE_PATH}
        ${SNPE_INCLUDE_PATH}
        ${EIGEN_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${UTILS_INCLUDE_PATH}
)

link_directories(
        ${OPENCV_LIB_PATH}
        ${ONNXRUNTIME_LIB_PATH}
        ${SNPE_LIB_PATH}
)
add_subdirectory(midas)
add_subdirectory(pfld)
add_subdirectory(retinaface)
add_subdirectory(yolov8)
add_subdirectory(yolov8face)
add_subdirectory(yolov8pose)
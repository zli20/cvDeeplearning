cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB UTILS_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h")
FILE(GLOB UTILS_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp")

FILE(GLOB SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/retinaface/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/yolov8/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/yolov8pose/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pfld/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/midas/*.cpp")

FILE(GLOB HEAD_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}retinaface/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/yolov8/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/yolov8pose/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/pfld/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/midas/*.h")

#FILE(GLOB ONNXRUNTIME_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/snpe1.68/*.h")
#FILE(GLOB SNPE1.68_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/onnxruntime/*.h")
#FILE(GLOB ONNXRUNTIME_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/snpe1.68/*.cpp")
#FILE(GLOB SNPE1.68_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/asset/onnxruntime/*.cpp")

message(STATUS "OpenCV_INCLUDE_DIRS = ${OPENCV_INCLUDE_PATH}")


include_directories(
        ${OPENCV_INCLUDE_PATH}
        ${ONNXRUNTIME_INCLUDE_PATH}
        ${SNPE_INCLUDE_PATH}
)

link_directories(
        ${OPENCV_LIB_PATH}
        ${ONNXRUNTIME_LIB_PATH}
        ${SNPE_LIB_PATH}
)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/
        ${CMAKE_CURRENT_SOURCE_DIR}/retinaface/
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8/
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8pose/
        ${CMAKE_CURRENT_SOURCE_DIR}/pfld/
        ${CMAKE_CURRENT_SOURCE_DIR}/midas/
)

#add_executable(cvDL ${UTILS_HEAD_FILES} ${UTILS_SRC_FILES} ${SRC_FILES} ${HEAD_FILES})
#target_link_libraries(cvDL ${OPENCV_LIBS} ${SNPE_LIBS} ${ONNXRUNTIME_LIBS} )
add_subdirectory(midas)
add_subdirectory(pfld)
add_subdirectory(retinaface)
add_subdirectory(yolov8)

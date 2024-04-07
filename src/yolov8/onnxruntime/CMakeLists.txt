set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

set(SOURCE_FILES
        Yolov8Onnx.cpp
        Yolov8Onnx.h
        main.cpp
        ../../utils/Datatype.h
)

add_executable(yolov8_onnx ${SOURCE_FILES} ${UTILS_SRC_FILES})

target_link_libraries(yolov8_onnx ${SNPE_LIBS})
target_link_libraries(yolov8_onnx ${ONNXRUNTIME_LIBS})
target_link_libraries(yolov8_onnx ${OPENCV_LIBS})

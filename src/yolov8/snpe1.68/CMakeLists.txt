set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
        ${OPENCV_LIB_PATH}
        ${ONNXRUNTIME_LIB_PATH}
        ${SNPE_LIB_PATH}
)

set(YOLOV8_DET_SOURCE_SNPE
        Yolov8DetSnpe.cpp
        Yolov8DetSnpe.h
        main.cpp
)

add_executable(yolov8_det_snpe ${YOLOV8_DET_SOURCE_SNPE} ${UTILS_SRC_FILES})

target_link_libraries(yolov8_det_snpe ${SNPE_LIBS})
target_link_libraries(yolov8_det_snpe ${ONNXRUNTIME_LIBS})
target_link_libraries(yolov8_det_snpe ${OPENCV_LIBS})

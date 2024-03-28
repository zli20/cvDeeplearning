set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OPENCV_INCLUDE_PATH}")

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
        Yolov8FaceSnpe.cpp
        Yolov8FaceSnpe.h
        main.cpp
)

add_executable(yolov8_face_snpe ${SOURCE_FILES} ${UTILS_SRC_FILES})

target_link_libraries(yolov8_face_snpe ${SNPE_LIBS})
target_link_libraries(yolov8_face_snpe ${ONNXRUNTIME_LIBS})
target_link_libraries(yolov8_face_snpe ${OPENCV_LIBS})

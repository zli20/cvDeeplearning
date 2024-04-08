set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SNPE_ENGINE_INCLUDE_PATH}
)

link_directories(
        ${OPENCV_LIB_PATH}
        ${ONNXRUNTIME_LIB_PATH}
        ${SNPE_LIB_PATH}
)

set(YOLOV8_FACE_SOURCE_SNPE
        Yolov8FaceSnpe.cpp
        Yolov8FaceSnpe.h
)

add_executable(yolov8_face_snpe ${YOLOV8_FACE_SOURCE_SNPE} ${UTILS_SRC_FILES} ${SNPE_ENGINE_SRC_FILES} main.cpp)

target_link_libraries(yolov8_face_snpe ${SNPE_LIBS})
target_link_libraries(yolov8_face_snpe ${OPENCV_LIBS})

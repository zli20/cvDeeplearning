set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SNPE_ENGINE_INCLUDE_PATH}
)


set(YOLOV8_POSE_SOURCE_SNPE
        Yolov8PoseSnpe.cpp
        Yolov8PoseSnpe.h
        main.cpp
)

add_executable(yolov8_pose_snpe ${YOLOV8_POSE_SOURCE_SNPE} ${UTILS_SRC_FILES} ${SNPE_ENGINE_SRC_FILES})

target_link_libraries(yolov8_pose_snpe ${SNPE_LIBS})
target_link_libraries(yolov8_pose_snpe ${OPENCV_LIBS})

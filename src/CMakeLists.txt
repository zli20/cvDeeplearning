cmake_minimum_required(VERSION 3.10)


FILE(GLOB UTILS_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/utils/*.h")
FILE(GLOB UTILS_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/utils/*.cpp")
set(UTILS_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/engine/utils/")

FILE(GLOB ONNXRUNTIME_ENGINE_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/onnxEngine/*.h")
FILE(GLOB ONNXRUNTIME_ENGINEE_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/onnxEngine/*.cpp")
set(ONNXRUNTIME_ENGINE_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/engine/onnxEngine/")

FILE(GLOB SNPE_ENGINE_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/snpeEngine/*.h")
FILE(GLOB SNPE_ENGINE_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/snpeEngine/*.cpp")
set(SNPE_ENGINE_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/engine/snpeEngine/")

FILE(GLOB BYTETRACK_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tracker/bytetrack/*.h")
FILE(GLOB BYTETRACK_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tracker/bytetrack/*.cpp")
set(BYTETRACK_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tracker/bytetrack/")
#add_library(bytetrack ${BYTETRACK_SRC_FILES})
#target_include_directories(bytetrack PUBLIC ${BYTETRACK_INCLUDE_PATH})

FILE(GLOB DEEPSORT_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tracker/deepsort/*.h")
FILE(GLOB DEEPSORT_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tracker/deepsort/*.cpp")
set(DEEPSORT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tracker/deepsort/")
#add_library(deepsort ${DEEPSORT_SRC_FILES})
#target_include_directories(deepsort PUBLIC ${DEEPSORT_INCLUDE_PATH})

FILE(GLOB MULTRACK_HEAD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/mulTrackEngine/*.h")
FILE(GLOB MULTRACK_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/engine/mulTrackEngine/*.cpp")
set(MULTRACK_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/engine/mulTrackEngine/")
#add_library(mulTrackEngine ${MULTRACK_SRC_FILES})
#target_include_directories(mulTrackEngine PUBLIC ${MULTRACK_INCLUDE_PATH})
#target_link_libraries(mulTrackEngine PUBLIC bytetrack deepsort)

include_directories(
        ${OPENCV_INCLUDE_PATH}
        ${SNPE_INCLUDE_PATH}
        ${ONNXRUNTIME_INCLUDE_PATH}
        ${EIGEN_INCLUDE_PATH}
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
#add_subdirectory(tracker)
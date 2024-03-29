set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(SOURCE_FILES_RETINAFACE
        RetinafaceSnpe.h
        RetinafaceSnpe.cpp
)

add_executable(retinaface_snpe ${SOURCE_FILES_RETINAFACE} ${UTILS_SRC_FILES} main.cpp)

target_link_libraries(retinaface_snpe ${SNPE_LIBS})
target_link_libraries(retinaface_snpe ${ONNXRUNTIME_LIBS})
target_link_libraries(retinaface_snpe ${OPENCV_LIBS})

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SNPE_ENGINE_INCLUDE_PATH}
)

set(SOURCE_FILES_MIDAS
        MidasSnpe.cpp
        MidasSnpe.h
)

add_executable(midas_snpe ${SOURCE_FILES_MIDAS} ${UTILS_SRC_FILES} ${SNPE_ENGINE_SRC_FILES} main.cpp)

target_link_libraries(midas_snpe ${SNPE_LIBS})
target_link_libraries(midas_snpe ${ONNXRUNTIME_LIBS})
target_link_libraries(midas_snpe ${OPENCV_LIBS})

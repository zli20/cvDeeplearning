set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../retinaface/snpe1.68/
)

set(SOURCE_FILES_PFLD
        PfldSnpe.cpp
        PfldSnpe.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../../retinaface/snpe1.68/RetinafaceSnpe.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../../retinaface/snpe1.68/RetinafaceSnpe.cpp
)

add_executable(pfld_snpe ${SOURCE_FILES_PFLD} ${UTILS_SRC_FILES} main.cpp)

target_link_libraries(pfld_snpe ${SNPE_LIBS})
target_link_libraries(pfld_snpe ${ONNXRUNTIME_LIBS})
target_link_libraries(pfld_snpe ${OPENCV_LIBS})

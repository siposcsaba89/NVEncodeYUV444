cmake_minimum_required(VERSION 3.0)


project(NV_Encode)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(NvEncoder)

set(SOURCES
    main.cpp
)

cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ::NvHWEncoder ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(compare compare.cpp)
target_link_libraries(compare ${OpenCV_LIBRARIES} ::NvHWEncoder ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib)
target_include_directories(compare PRIVATE ${OpenCV_INCLUDE_DIRS})
cmake_minimum_required(VERSION 3.0)
project(NvHWEncoder)


find_package(CUDA REQUIRED)

set(PUBLIC_HEADERS 
    include/NvEncoder/nvFileIO.h
    include/NvEncoder/nvCPUOPSys.h
    include/NvEncoder/nvUtils.h
    include/NvEncoder/nvEncodeAPI.h
    include/NvEncoder/NvEncoder.h
    include/NvEncoder/NvHWEncoder.h)

set(SRCS
    NvHWEncoder.cpp
    NvEncoder.cpp
)    
add_library(${PROJECT_NAME} ${PUBLIC_HEADERS} ${SRCS})
add_library(::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include
        ${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})
  
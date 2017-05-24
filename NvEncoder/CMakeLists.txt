cmake_minimum_required(VERSION 3.0)
project(nvencoder)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED 1)

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
        $<INSTALL_INTERFACE:include>
        ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    CXX_STANDARD 11 
    CXX_STANDARD_REQUIRED 1
    DEBUG_POSTFIX _d
    FOLDER libs)

install(
	FILES
        ${PUBLIC_HEADERS}
    DESTINATION
        include/NvEncoder/
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/nvencoder/nvencoder-config-version.cmake"
  VERSION 0.0.1
  COMPATIBILITY AnyNewerVersion
)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvencoder-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/nvencoder/nvencoder-config.cmake"
  @ONLY
)


install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/nvencoder/nvencoder-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/nvencoder/nvencoder-config-version.cmake"
  DESTINATION
    lib/cmake/nvencoder
)

install(TARGETS nvencoder EXPORT nvencoder-targets
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(EXPORT nvencoder-targets DESTINATION lib/cmake/nvencoder NAMESPACE ::)


########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")
set_target_properties(uninstall PROPERTIES FOLDER CMakePredefinedTargets)
  
cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(cxxLCHF_pybind)

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -fPIC -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fPIC -Wno-sign-compare")

set(CMAKE_PREFIX_PATH /home/meiqua/anaconda3/envs/furnace)
# we install cuda9 but build opencv with cuda8, have to set mannually
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0)
# we install opencv in our conda env
#set(OpenCV_DIR /home/meiqua/anaconda3/envs/furnace/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
#message(STATUS "dir='${OpenCV_DIR}'")

find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)

include(Cmake/FindNumPy.cmake)

include_directories(${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_PYBIND
    pybind11.cpp
    np2mat/ndarray_converter.cpp)

set(SOURCE
    lchf.cpp
    forest.cpp
    meanshift/MeanShift.cpp)
set(LIBS
    ${OpenCV_LIBS}
    proto
    ${PROTOBUF_LIBRARY}
    stdc++fs)

ADD_SUBDIRECTORY(proto)

#set_target_properties(proto PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
pybind11_add_module(${PROJECT_NAME} ${SOURCE} ${SOURCE_PYBIND})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

add_executable(${PROJECT_NAME}_executable ${SOURCE} test.cpp)
target_link_libraries(${PROJECT_NAME}_executable ${LIBS})

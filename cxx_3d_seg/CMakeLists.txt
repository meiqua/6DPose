cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(cxx_3d_seg_pybind)

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -fPIC")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fPIC")

set(CMAKE_PREFIX_PATH /home/meiqua/anaconda3/envs/furnace)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0)
find_package(OpenCV 3 REQUIRED)
#message(STATUS "dir='${OpenCV_DIR}'")

find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)
include(Cmake/FindNumPy.cmake)

include_directories(${INCLUDE_DIRECTORIES} 
    "/usr/include/eigen3"
    "asp/include"
    "asp")

include_directories(${INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS} 
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_PYBIND
    pybind11.cpp
    np2mat/ndarray_converter.cpp)

set(SOURCE_ASP
	asp/src/libasp/algos/ASP.cpp
	asp/src/libasp/algos/DASP.cpp
	asp/src/libasp/algos/SLIC.cpp
	asp/src/libasp/pds/pds.cpp
	asp/src/libasp/pds/Grid.cpp
	asp/src/libasp/pds/FloydSteinberg.cpp
)

set(SOURCE
    ${SOURCE_ASP}
    cxx_3d_seg.cpp)

set(LIBS
    ${OpenCV_LIBS})

pybind11_add_module(${PROJECT_NAME} ${SOURCE} ${SOURCE_PYBIND})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

add_executable(${PROJECT_NAME}_executable ${SOURCE} test.cpp)
target_link_libraries(${PROJECT_NAME}_executable ${LIBS})


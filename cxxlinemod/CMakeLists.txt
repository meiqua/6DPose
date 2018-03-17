cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(cxxlinemod)

# we install cuda9 but build opencv with cuda8, have to set mannually
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0)
# we install opencv in our conda env
set(OpenCV_DIR /home/meiqua/anaconda3/envs/furnace/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
#message(STATUS "dir='${OpenCV_LIBS}'")

find_package(pybind11 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
pybind11_add_module(${PROJECT_NAME} ${PROJECT_NAME}.cpp pybind11.cpp)

add_executable(${PROJECT_NAME}_executable ${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME}_executable ${OpenCV_LIBS})


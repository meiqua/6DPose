#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>

void show_image(cv::Mat image);

cv::Mat read_image(std::string image_name);

cv::Mat passthru(cv::Mat image);

cv::Mat cloneimg(cv::Mat image);

#endif

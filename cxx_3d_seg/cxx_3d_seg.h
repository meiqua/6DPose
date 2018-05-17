#ifndef CXX_3D_SEG_H
#define CXX_3D_SEG_H

#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <slimage/opencv.hpp>
#include <slimage/algorithm.hpp>
#include <asp/algos.hpp>

#include "super4pcs/algorithms/super4pcs.h"
#include "super4pcs/io/io.h"
#include <opencv2/rgbd.hpp>

// seg cloud to convex part, sorted by pixel counts
cv::Mat convex_cloud_seg(cv::Mat& rgb, cv::Mat& depth, cv::Mat& sceneK);

cv::Mat depth2cloud(cv::Mat& depth, cv::Mat& mask, cv::Mat& sceneK);

cv::Mat pose_estimation(cv::Mat& cloud, std::string ply_model,
                        float LCP_thresh = 0.5, int pcs_secends = 1);

#endif

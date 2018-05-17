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
#include "linemod_icp.h"
// seg cloud to convex part, sorted by pixel counts
cv::Mat convex_cloud_seg(cv::Mat& rgb, cv::Mat& depth, cv::Mat& sceneK);

cv::Mat depth2cloud(cv::Mat& depth, cv::Mat& mask, cv::Mat& sceneK);

cv::Mat pose_estimation(cv::Mat& cloud, std::string ply_model,
                        float LCP_thresh = 0.5, float ICP_thresh = 100,
                        bool use_pcs = true, int pcs_seconds=1,
                        bool use_icp = false, int cloud_icp_size = 1000, int model_icp_size = 10000);

#endif

// originally running linemod fails under py,
// this file is designed for running in c++ then binding to py
// however after some effort I find out that depth type matters
// now linemod works fine under py

#include "cxxlinemod.h"
#include <memory>
#include <iostream>
#include <opencv2/surface_matching.hpp>
using namespace std;
using namespace cv;
cv::Mat depth2pc(cv::Mat depth, cv::Mat K)
{
     cv::Mat pc;
//     auto mask = depth>0;
     cv::rgbd::depthTo3d(depth, K, pc);
     return pc;
}



// for test
int main(){
    Mat depth = cv::imread("/home/meiqua/6DPose/public/datasets/hinterstoisser/test/09/depth/0000.png", IMREAD_ANYDEPTH);

    Mat K = (Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
//    Mat pc = depth2pc(depth, K);
    auto pr = std::make_unique<poseRefine>();
    pr->process(depth, depth, K, K, depth, depth, 0, 0);

    cout << "break point line" << endl;
    return 0;
}


Mat poseRefine::process(Mat &sceneDepth, Mat &modelDepth, Mat &sceneK, Mat &modelK,
                        Mat &modelR, Mat &modelT, int detectX, int detectY)
{
    cv::Mat modelCloudWithNorm, sceneCloudWithNorm;  //rows x cols x 6

    modelCloudWithNorm = normalCompute(modelDepth, modelK);
    sceneCloudWithNorm = normalCompute(sceneDepth, sceneK);

    auto poseInit = make_unique<cv::ppf_match_3d::Pose3D>();

    // well, I can't find a better method
    auto m33 = cv::Matx33d(modelR.at<float>(0, 0), modelR.at<float>(0, 1), modelR.at<float>(0, 2),
                       modelR.at<float>(1, 0), modelR.at<float>(1, 1), modelR.at<float>(1, 2),
                       modelR.at<float>(2, 0), modelR.at<float>(2, 1), modelR.at<float>(2, 2));
    auto v3d = cv::Vec3d(modelT.at<float>(0, 0), modelT.at<float>(1, 0), modelT.at<float>(2, 0));
    poseInit->updatePose(m33, v3d);
    auto icp = make_unique<cv::ppf_match_3d::ICP>(10);
//    icp.registerModelToScene(modelCloudWithNorm, sceneCloudWithNorm);

    return modelCloudWithNorm;
}

float poseRefine::getConfidence()
{
    return confidence;
}

Mat poseRefine::normalCompute(Mat & depth, Mat &K)
{
    cv::Mat modelCloud;  //rows x cols x 3
    cv::rgbd::depthTo3d(depth, K, modelCloud);

    cv::Mat modelMask = depth > 0;
    int modelCloudNum = cv::countNonZero(modelMask);

    auto normal_computer = cv::rgbd::RgbdNormals(depth.rows, depth.cols, CV_32F, K);
    cv::Mat modelNormals;
    normal_computer(modelCloud, modelNormals);

    cv::Mat modelCloudWithNorm(modelCloudNum, 6, CV_32FC1);
    auto modelIt = modelCloud.begin<cv::Vec3f>();
    auto normIt = modelCloud.begin<cv::Vec3f>();
    auto maskIt = depth.begin<float>();
    for(auto i=0; modelIt != modelCloud.begin<cv::Vec3f>(); modelIt++, normIt++, maskIt++){
        if(*maskIt > 0){
            auto cloud3f = *modelIt;
            auto norm3f = *normIt;
            for(auto j=0; j<3; j++){
                modelCloudWithNorm.at<float>(i, j) = cloud3f[j];
                modelCloudWithNorm.at<float>(i, j+3) = norm3f[j];
            }
            i++;
        }
    }
    return modelCloudWithNorm;
}

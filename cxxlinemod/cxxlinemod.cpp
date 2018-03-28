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
    cv::Mat modelCloud;  //rows x cols x 3
    cv::rgbd::depthTo3d(modelDepth, modelK, modelCloud);
    cv::Mat modelMask = modelDepth > 0;

    cv::Mat sceneCloud;  //rows x cols x 3
    cv::rgbd::depthTo3d(sceneDepth, sceneK, sceneCloud);
    cv::Mat sceneMask = sceneDepth > 0;

    // modelT xy is the xy of point in the scene at bbox center
    Mat non0p;
    findNonZero(modelMask,non0p);
    Rect bbox=boundingRect(non0p);
    int offsetX = bbox.width/2 + detectX;
    int offsetY = bbox.height/2 + detectY;

    int smoothSize = 5;
    //boundary check
    int startoffsetX1 = offsetX - smoothSize/2;
    if(startoffsetX1 < 0) startoffsetX1 = 0;
    int startoffsetX2 = offsetX + smoothSize/2;
    if(startoffsetX2 > sceneCloud.cols) startoffsetX2 = sceneCloud.cols;
    int startoffsetY1 = offsetY - smoothSize/2;
    if(startoffsetY1 < 0) startoffsetY1 = 0;
    int startoffsetY2 = offsetY + smoothSize/2;
    if(startoffsetY2 > sceneCloud.rows) startoffsetY2 = sceneCloud.rows;

    cv::Vec3f avePoint; int count=0;
    for(auto i=startoffsetX1; i<startoffsetX2; i++){
        for(auto j=startoffsetY1; j<startoffsetY2; j++){
            auto p = sceneCloud.at<cv::Vec3f>(j, i);
            avePoint += p;
            count++;
        }
    }
    avePoint /= count;
    modelT.at<float>(0, 0) = avePoint[0];
    modelT.at<float>(1, 0) = avePoint[1];
    avePoint[2] = 0;  //  don't transfer modelCloud's z

    auto modelIt = modelCloud.begin<cv::Vec3f>();
    for(; modelIt != modelCloud.begin<cv::Vec3f>(); modelIt++){
        *modelIt += avePoint;
    }

    cv::Mat modelCloudWithNorm, sceneCloudWithNorm;  //rows x cols x 6
    modelCloudWithNorm = normalCompute(modelCloud, modelK, modelMask);
    sceneCloudWithNorm = normalCompute(sceneCloud, sceneK, sceneMask);

    //poseInit has been applied to model
    auto poseInit = make_unique<cv::ppf_match_3d::Pose3D>();
    // well, it looks stupid
    auto m33 = cv::Matx33d(modelR.at<float>(0, 0), modelR.at<float>(0, 1), modelR.at<float>(0, 2),
                       modelR.at<float>(1, 0), modelR.at<float>(1, 1), modelR.at<float>(1, 2),
                       modelR.at<float>(2, 0), modelR.at<float>(2, 1), modelR.at<float>(2, 2));
    auto v3d = cv::Vec3d(modelT.at<float>(0, 0), modelT.at<float>(1, 0), modelT.at<float>(2, 0));
    poseInit->updatePose(m33, v3d);

    auto icp = make_unique<cv::ppf_match_3d::ICP>(10); //10 iteration
    cv::Matx44d pose;
    icp->registerModelToScene(modelCloudWithNorm, sceneCloudWithNorm, residual, pose);
    poseInit->appendPose(pose);
    return Mat(4, 4, CV_32FC1, &poseInit->pose);
}

double poseRefine::getResidual()
{
    return residual;
}

Mat poseRefine::normalCompute(Mat &cloud, Mat &K, Mat &mask)
{
    int modelCloudNum = cv::countNonZero(mask);

    auto normal_computer = cv::rgbd::RgbdNormals(mask.rows, mask.cols, CV_32F, K);
    cv::Mat modelNormals;
    normal_computer(cloud, modelNormals);

    cv::Mat modelCloudWithNorm(modelCloudNum, 6, CV_32FC1);
    auto modelIt = cloud.begin<cv::Vec3f>();
    auto normIt = modelNormals.begin<cv::Vec3f>();
    auto maskIt = mask.begin<float>();
    for(auto i=0; modelIt != cloud.begin<cv::Vec3f>(); modelIt++, normIt++, maskIt++){
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


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
    // test case1
    Mat depth = cv::imread("/home/meiqua/6DPose/cxxlinemod/test/case1/0003.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    Mat depth_ren = cv::imread("/home/meiqua/6DPose/cxxlinemod/test/case1/depth_ren.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    Mat K = (Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
    Mat R = (Mat_<float>(3,3) << 1.00000000, 0.00000000, 0.00000000,
             0.00000000, -0.90727223, -0.42054381, 0.00000000, 0.42054381, -0.90727223);
    Mat t = (Mat_<float>(3,1) << 0.0, 0.0, 1000.0);
    auto pr = std::make_unique<poseRefine>();
    cv::Mat pose = pr->process(depth, depth_ren, K, K, R, t, 352, 327);

    cout << "M = "<< endl << " "  << pose << endl << endl;

    cout << "break point line" << endl;
    return 0;
}

Mat poseRefine::process(Mat &sceneDepth, Mat &modelDepth, Mat &sceneK, Mat &modelK,
                        Mat &modelR, Mat &modelT, int detectX, int detectY)
{
//    double unit = 1.0;
//    sceneDepth.convertTo(sceneDepth, CV_32F);
//    modelDepth.convertTo(modelDepth, CV_32F);
//    sceneK.convertTo(sceneK, CV_32F);
//    modelK.convertTo(modelK, CV_32F);
//    modelR.convertTo(modelR, CV_32F);
//    modelT.convertTo(modelT, CV_32F);
//    sceneDepth /= unit;
//    modelDepth /= unit;
//    sceneK /= unit;
//    modelK /= unit;
//    modelR /= unit;
//    modelT /= unit;

    cv::Mat modelMask = modelDepth > 0;
    Mat non0p;
    findNonZero(modelMask,non0p);
    Rect bbox=boundingRect(non0p);

    cv::Mat modelCloud_cropped;  //rows x cols x 3, cropped
    cv::Mat modelDepth_cropped = modelDepth(bbox);
    cv::Mat modelMask_cropped = modelMask(bbox);
    cv::rgbd::depthTo3d(modelDepth_cropped, modelK, modelCloud_cropped);

    //crop scene
    int enhancedX = bbox.width/2;
    int enhancedY = bbox.height/2;
    //boundary check
    int bboxX1 = detectX - enhancedX;
    if(bboxX1 < 0) bboxX1 = 0;
    int bboxX2 = detectX + bbox.width + enhancedX;
    if(bboxX2 > sceneDepth.cols) bboxX2 = sceneDepth.cols;
    int bboxY1 = detectY - enhancedY;
    if(bboxY1 < 0) bboxY1 = 0;
    int bboxY2 = detectY + bbox.height + enhancedY;
    if(bboxY2 > sceneDepth.rows) bboxY1 = sceneDepth.rows;

    cv::Rect ROI_sceneDepth(bboxX1, bboxY1, bboxX2-bboxX1, bboxY2-bboxY1);
    cv::Mat sceneDepth_cropped = sceneDepth(ROI_sceneDepth);
    cv::Mat sceneMask_cropped = sceneDepth_cropped > 0;
    cv::Mat sceneCloud_cropped;
    cv::rgbd::depthTo3d(sceneDepth_cropped, sceneK, sceneCloud_cropped);

    cv::Mat modelCloudWithNorm, sceneCloudWithNorm;  //rows x cols x 6
    modelCloudWithNorm = normalCompute(modelCloud_cropped, modelK, modelMask_cropped);
    sceneCloudWithNorm = normalCompute(sceneCloud_cropped, sceneK, sceneMask_cropped);

    auto icp = make_unique<cv::ppf_match_3d::ICP>(10); //10 iteration
    cv::Matx44d pose;
    icp->registerModelToScene(modelCloudWithNorm, sceneCloudWithNorm, residual, pose);

    //poseInit has been applied to model
    auto poseInit = make_unique<cv::ppf_match_3d::Pose3D>();

    int smoothSize = 5;
    //boundary check
    int offsetX = sceneDepth_cropped.cols/2;
    int offsetY = sceneDepth_cropped.rows/2;
    int startoffsetX1 = offsetX - smoothSize/2;
    if(startoffsetX1 < 0) startoffsetX1 = 0;
    int startoffsetX2 = offsetX + smoothSize/2;
    if(startoffsetX2 > sceneDepth_cropped.cols) startoffsetX2 = sceneDepth_cropped.cols;
    int startoffsetY1 = offsetY - smoothSize/2;
    if(startoffsetY1 < 0) startoffsetY1 = 0;
    int startoffsetY2 = offsetY + smoothSize/2;
    if(startoffsetY2 > sceneDepth_cropped.rows) startoffsetY2 = sceneDepth_cropped.rows;

    cv::Vec3f avePoint; int count=0;
    for(auto i=startoffsetX1; i<startoffsetX2; i++){
        for(auto j=startoffsetY1; j<startoffsetY2; j++){
            auto p = sceneDepth_cropped.at<cv::Vec3f>(j, i);
            avePoint += p;
            count++;
        }
    }
    avePoint /= count;
    modelT.at<float>(0, 0) = avePoint[0];
    modelT.at<float>(1, 0) = avePoint[1];
    // well, it looks stupid
    auto m33 = cv::Matx33d(modelR.at<float>(0, 0), modelR.at<float>(0, 1), modelR.at<float>(0, 2),
                       modelR.at<float>(1, 0), modelR.at<float>(1, 1), modelR.at<float>(1, 2),
                       modelR.at<float>(2, 0), modelR.at<float>(2, 1), modelR.at<float>(2, 2));
    auto v3d = cv::Vec3d(modelT.at<float>(0, 0), modelT.at<float>(1, 0), modelT.at<float>(2, 0));
    poseInit->updatePose(m33, v3d);
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


// originally running linemod fails under py,
// this file is designed for running in c++ then binding to py
// however after some effort I find out that depth type matters
// now linemod works fine under py

#include "cxxlinemod.h"
#include <memory>
#include <iostream>
#include <opencv2/surface_matching.hpp>
#include "linemod_icp.h"
#include <assert.h>
using namespace std;
using namespace cv;

// for test
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

static std::string prefix = "/home/meiqua/6DPose/cxxlinemod/test/case1/";
// use rgb to check ROI
static Mat rgb = cv::imread(prefix+"rgb.png");
static Mat rgb_ren = cv::imread(prefix+"rgb_ren.png");
int main(){
    // test case1
    /*
- cam_R_m2c: [0.99710798, -0.07587780, -0.00426525,
             -0.06000210, -0.75155902, -0.65693200,
              0.04664090, 0.65528798, -0.75393802]
  cam_t_m2c: [100.61034025, 180.32773127, 1024.07664363]
  obj_bb: [352, 324, 57, 48]
  */

    Mat depth = cv::imread(prefix+"0003.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    Mat depth_ren = cv::imread(prefix+"depth_ren.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
//    cout << "depth: " << type2str(depth.type())  << endl;

    Mat K = (Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
    Mat R = (Mat_<float>(3,3) << 1.00000000, 0.00000000, 0.00000000,
             0.00000000, -0.90727223, -0.42054381, 0.00000000, 0.42054381, -0.90727223);
    Mat t = (Mat_<float>(3,1) << 0.0, 0.0, 1000.0);
    auto pr = std::make_unique<poseRefine>();
    pr->process(depth, depth_ren, K, K, R, t, 352, 327);

    cout << pr->getR() << endl;
    cout << pr->getT() << endl;
    cout << pr->getResidual() << endl;

    cout << "break point line" << endl;
    return 0;
}

void poseRefine::process(Mat &sceneDepth, Mat &modelDepth, Mat &sceneK, Mat &modelK,
                        Mat &modelR, Mat &modelT, int detectX, int detectY)
{
//    sceneDepth.convertTo(sceneDepth, CV_16U);
//    modelDepth.convertTo(modelDepth, CV_16U);
    assert(sceneDepth.type() == CV_16U);
    assert(sceneK.type() == CV_32F);

    cv::Mat modelMask = modelDepth > 0;
    Mat non0p;
    findNonZero(modelMask,non0p);
    Rect bbox=boundingRect(non0p);

    //crop scene
    int enhancedX = bbox.width/8*0;
    int enhancedY = bbox.height/8*0; // no padding
    //boundary check
    int bboxX1 = detectX - enhancedX;
    if(bboxX1 < 0) bboxX1 = 0;
    int bboxX2 = detectX + bbox.width + enhancedX;
    if(bboxX2 > sceneDepth.cols) bboxX2 = sceneDepth.cols;
    int bboxY1 = detectY - enhancedY;
    if(bboxY1 < 0) bboxY1 = 0;
    int bboxY2 = detectY + bbox.height + enhancedY;
    if(bboxY2 > sceneDepth.rows) bboxY2 = sceneDepth.rows;

    int bboxX1_ren = bbox.x - enhancedX;
    if(bboxX1_ren < 0) bboxX1_ren = 0;
    int bboxX2_ren = bbox.x + bbox.width + enhancedX;
    if(bboxX2_ren > sceneDepth.cols) bboxX2_ren = sceneDepth.cols;
    int bboxY1_ren = bbox.y - enhancedY;
    if(bboxY1_ren < 0) bboxY1_ren = 0;
    int bboxY2_ren = bbox.y + bbox.height + enhancedY;
    if(bboxY2_ren > sceneDepth.rows) bboxY2_ren = sceneDepth.rows;

    cv::Rect ROI_sceneDepth(bboxX1, bboxY1, bboxX2-bboxX1, bboxY2-bboxY1);
    cv::Rect ROI_modelDepth(bboxX1_ren, bboxY1_ren, bboxX2_ren-bboxX1_ren, bboxY2_ren-bboxY1_ren);
    cv::Mat modelCloud_cropped;  //rows x cols x 3, cropped
    cv::Mat modelDepth_cropped = modelDepth(ROI_modelDepth);
    cv::rgbd::depthTo3d(modelDepth_cropped, modelK, modelCloud_cropped);

    cv::Mat sceneDepth_cropped = sceneDepth(ROI_sceneDepth);
    cv::Mat sceneCloud_cropped;
    cv::rgbd::depthTo3d(sceneDepth_cropped, sceneK, sceneCloud_cropped);
//    imshow("rgb_ren_cropped", rgb_ren(ROI_modelDepth));
//    imshow("rgb_cropped", rgb(ROI_sceneDepth));
//    waitKey(1000000);

    // get x,y coordinate of obj in scene
    // previous depth-cropped-first version is for icp
    // cropping depth first means we move ROI cloud to center of view
    cv::Mat sceneCloud;
    cv::rgbd::depthTo3d(sceneDepth, sceneK, sceneCloud);

    int smoothSize = 7;
    //boundary check
    int offsetX = ROI_sceneDepth.x + ROI_sceneDepth.width/2;
    int offsetY = ROI_sceneDepth.y + ROI_sceneDepth.height/2;
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
            if(checkRange(p)){
                avePoint += p;
                count++;
            }
        }
    }
    avePoint /= count;
    modelT.at<float>(0, 0) = avePoint[0]*1000; // scene cloud unit is meter
    modelT.at<float>(1, 0) = avePoint[1]*1000;
    // well, it looks stupid
    auto R_real_icp = cv::Matx33f(modelR.at<float>(0, 0), modelR.at<float>(0, 1), modelR.at<float>(0, 2),
                       modelR.at<float>(1, 0), modelR.at<float>(1, 1), modelR.at<float>(1, 2),
                       modelR.at<float>(2, 0), modelR.at<float>(2, 1), modelR.at<float>(2, 2));
    auto T_real_icp = cv::Vec3f(modelT.at<float>(0, 0), modelT.at<float>(1, 0), modelT.at<float>(2, 0));

    std::vector<cv::Vec3f> pts_real_model_temp;
    std::vector<cv::Vec3f> pts_real_ref_temp;
    float px_ratio_missing = matToVec(sceneCloud_cropped, modelCloud_cropped, pts_real_ref_temp, pts_real_model_temp);

    float px_ratio_match_inliers = 0.0f;
    float icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp,
                                     T_real_icp, px_ratio_match_inliers, 1);

    icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp,
                               T_real_icp, px_ratio_match_inliers, 2);

    icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp,
                               T_real_icp, px_ratio_match_inliers, 0);
    R_refined = Mat(R_real_icp);
    t_refiend = Mat(T_real_icp);
    residual = icp_dist;
}

float poseRefine::getResidual()
{
    return residual;
}

Mat poseRefine::getR()
{
    return R_refined;
}

Mat poseRefine::getT()
{
    return t_refiend;
}


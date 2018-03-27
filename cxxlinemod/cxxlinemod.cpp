// originally running linemod fails under py,
// this file is designed for running in c++ then binding to py
// however after some effort I find out that depth type matters
// now linemod works fine under py

#include "cxxlinemod.h"
#include <iostream>
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
    Mat pc = depth2pc(depth, K);

    cout << "break point line" << endl;
    return 0;
}




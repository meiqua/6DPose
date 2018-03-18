// originally running linemod fails under py,
// this file is designed for running in c++ then binding to py
// however after some effort I find out that depth type matters
// now linemod works fine under py

#include "cxxlinemod.h"
#include<boost/format.hpp>
using namespace std;
using namespace cv;
int main(){
    auto detector = cv::linemod::getDefaultLINEMOD();

    std::string path = "/home/meiqua/6DPose/public/datasets/hinterstoisser/train/02/";

    for(auto i=0; i<20; i++){
        auto rgbPath = path + "rgb/" + boost::str(boost::format("%04d.png")%i);
        auto depthPath = path + "depth/" + boost::str(boost::format("%04d.png")%i);

        auto rgb = cv::imread(rgbPath);
        auto depth = cv::imread(depthPath, CV_LOAD_IMAGE_ANYDEPTH);
//        depth.convertTo(depth, CV_32FC1);
        double min, max;
        minMaxIdx(depth, &min, &max);
        std::cout << max <<std::endl;
         cv::waitKey(5000);
        auto mask = depth > 0;

        std::vector<cv::Mat> sources(2);
        sources[0] = rgb;
        sources[1] = depth;

        auto visual = true;
        if(visual){
            cv::namedWindow("rgb");
            cv::imshow("rgb", rgb);
            cv::namedWindow("depth");
            cv::imshow("depth", depth/1000.0);
            cv::waitKey(1000);
        }

        auto success = detector->addTemplate(sources, "obj2", mask);
        std::cout << "try " << i << ": " << success <<std::endl;
    }



    return 0;
}


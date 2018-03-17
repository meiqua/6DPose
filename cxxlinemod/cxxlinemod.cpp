#include "cxxlinemod.h"
#include<boost/format.hpp>
int main(){
    auto detector = cv::linemod::getDefaultLINEMOD();

    std::string path = "/home/meiqua/6DPose/public/datasets/hinterstoisser/train/02";

    for(auto i=0; i<20; i++){
        auto rgbPath = path + "rgb/" + boost::str(boost::format("%04d")%i);
        auto depthPath = path + "depth/" + boost::str(boost::format("%04d")%i);

        auto rgb = cv::imread(rgbPath);
        auto depth = cv::imread(depthPath);
        auto mask = depth > 0;

        std::vector<cv::Mat> sources(2);
        sources[0] = rgb;
        sources[1] = depth;
        auto success = detector->addTemplate(sources, "obj2", mask);
        std::cout << "try " << i << ": " << success <<std::endl;
    }



    return 0;
}


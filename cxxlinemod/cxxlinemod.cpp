// originally running linemod fails under py,
// this file is designed for running in c++ then binding to py
// however after some effort I find out that depth type matters
// now linemod works fine under py

#include "cxxlinemod.h"
#include<boost/format.hpp>

void show_image(cv::Mat image)
{
    cv::imshow("image_from_Cpp", image);
    cv::waitKey(0);
}

cv::Mat read_image(std::string image_name)
{
    cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
    return image;
}

cv::Mat passthru(cv::Mat image)
{
    return image;
}

cv::Mat cloneimg(cv::Mat image)
{
    return image.clone();
}

// for test
int main(){

    return 0;
}




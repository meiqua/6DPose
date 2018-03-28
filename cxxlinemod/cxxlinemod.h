#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class poseRefine{
public:
    poseRefine(): residual(-1){}
    cv::Mat process(cv::Mat& sceneDepth, cv::Mat& modelDepth, cv::Mat& sceneK, cv::Mat& modelK,
                    cv::Mat& modelR, cv::Mat& modelT, int detectX, int detectY);
    double getResidual();
private:
    cv::Mat normalCompute(cv::Mat& cloud, cv::Mat& K, cv::Mat& mask);

    double residual;
};

#endif

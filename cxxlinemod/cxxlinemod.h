#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat depth2pc(cv::Mat depth, cv::Mat K);

class poseRefine{
public:
    poseRefine(): confidence(0){}
    cv::Mat process(cv::Mat& sceneDepth, cv::Mat& modelDepth, cv::Mat& sceneK, cv::Mat& modelK,
                    cv::Mat& modelR, cv::Mat& modelT, int detectX, int detectY);
    float getConfidence();
private:
    cv::Mat normalCompute(cv::Mat& depth, cv::Mat& K);

    float confidence;
};

#endif

#include "lchf.h"
#include <memory>
#include <chrono>
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

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(){
        double t = elapsed();
        std::cout << "elasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
int main(){
    string prefix = "/home/meiqua/6DPose/cxxLCHF/test/";
    Mat rgb = cv::imread(prefix+"0000_rgb.png");
    Mat depth = cv::imread(prefix+"0000_dep.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    Mat mask = cv::imread(prefix+"train_mask.png");
    cvtColor(mask, mask, cv::COLOR_RGB2GRAY);
    Mat Points;
    findNonZero(mask,Points);
    Rect Min_Rect=boundingRect(Points);

    Timer tmr;
    auto crop_rgb = rgb(Min_Rect);
    auto crop_depth = depth(Min_Rect);
    Linemod_feature lf1(crop_rgb, crop_depth), lf2(crop_rgb, crop_depth);
    lf1.constructEmbedding();
    lf2.constructEmbedding();
    float score = lf1.similarity(lf2);
    cout << score << endl;
    cout << "end line" << endl;
    return 0;
}

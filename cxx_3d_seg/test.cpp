#include "cxx_3d_seg.h"
#include <asp/algos.hpp>
#include <chrono>
using namespace std;
using namespace cv;
// for test
namespace test_helper {
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "  elasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
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
template<typename T>
std::vector<T> unique(const cv::Mat& input, bool sort = false)
{
    std::vector<T> out;
    for (int y = 0; y < input.rows; ++y)
    {
        auto row_ptr = input.ptr<T>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            T value = row_ptr[x];

            if ( std::find(out.begin(), out.end(), value) == out.end() )
                out.push_back(value);
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());

    return out;
}
}


int main(){
    string prefix = "/home/meiqua/6DPose/cxx_3d_seg/test/3/";
    Mat rgb = cv::imread(prefix+"rgb/0000.png");
    Mat depth = cv::imread(prefix+"depth/0000.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

//    pyrDown(rgb, rgb);
//    pyrDown(depth, depth);

    test_helper::Timer timer;

    auto rgb_slimage = slimage::ConvertToSlimage(rgb);
    auto dep_slimage = slimage::ConvertToSlimage(depth);
    slimage::Image3ub img_color = slimage::anonymous_cast<unsigned char,3>(rgb_slimage);
    slimage::Image1ui16 img_depth = slimage::anonymous_cast<uint16_t,1>(dep_slimage);

    auto test_group = asp::DsapGrouping(img_color, img_depth);
    Mat idxs = slimage::ConvertToOpenCv(test_group);

    timer.out("grouping");

    std::vector<int> unik = test_helper::unique<int>(idxs, true);
    std::map<int, Vec3b> color_map;
    for(auto idx: unik){
        auto color = Vec3b(rand()%255, rand()%255, rand()%255);
        color_map[idx] = color;
    }

    Mat show = Mat(idxs.size(), CV_8UC3);
    auto show_iter = show.begin<Vec3b>();
    for(auto idx_iter = idxs.begin<int>(); idx_iter<idxs.end<int>();idx_iter++, show_iter++){
        auto color = color_map.find(*idx_iter)->second;
        *show_iter = color;
    }

    imshow("show", show);
//    imshow("rgb", rgb);
    waitKey(0);
//    Mat rgb_ = slimage::ConvertToOpenCv(rgb_slimage);
//    Mat depth_ = slimage::ConvertToOpenCv(dep_slimage);

    cout << "end" << endl;
    return 0;
}

#include "lchf.h"
#include "forest.h"
#include <memory>
#include <chrono>
#include <assert.h>
#include <opencv2/dnn.hpp>

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
    void out(string message = ""){
        double t = elapsed();
        std::cout << message << "\telasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

static std::string prefix = "/home/meiqua/6DPose/cxxLCHF/test";

void dataset_test(){
    Timer time;
    string pre = "/home/meiqua/6DPose/public/datasets/hinterstoisser/train/09/";

    int train_size = 1000;
    vector<Linemod_feature> features;
    features.reserve(train_size*5);
    for(int i=0;i<train_size;i++){
        auto i_str = to_string(i);
        for(int pad=4-i_str.size();pad>0;pad--){
            i_str = '0'+i_str;
        }
        Mat rgb = cv::imread(pre+"rgb/"+i_str+".png");
        Mat depth = cv::imread(pre+"depth/"+i_str+".png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);


        Mat Points;
        findNonZero(depth>0,Points);
        Rect bbox=boundingRect(Points);

        vector<Rect> bbox_5half(5);
        bbox_5half[0] = bbox;
        bbox_5half[0].width /= 2;
        bbox_5half[0].height /= 2;
        bbox_5half[1] = bbox_5half[0];
        bbox_5half[1].x = bbox_5half[0].x + bbox_5half[0].width;
        bbox_5half[1].y = bbox_5half[0].y;
        bbox_5half[2] = bbox_5half[0];
        bbox_5half[2].y = bbox_5half[0].y + bbox_5half[0].height;
        bbox_5half[2].x = bbox_5half[0].x;
        bbox_5half[3] = bbox_5half[0];
        bbox_5half[3].x = bbox_5half[0].x + bbox_5half[0].width;
        bbox_5half[3].y = bbox_5half[0].y + bbox_5half[0].height;
        bbox_5half[4] = bbox_5half[0];
        bbox_5half[4].x = bbox_5half[0].x + bbox_5half[0].width/2;
        bbox_5half[4].y = bbox_5half[0].y + bbox_5half[0].height/2;

        for(auto& aBox: bbox_5half){
            features.emplace_back(rgb(aBox), depth(aBox));
            features.back().constructEmbedding();
        }
//        cout << "features " << i << " OK" << endl;
//        cout << endl;
    }
    time.out("construct features");

    lchf_model model;
    model.train(features);
    time.out("train time:");

    model.path = prefix;
    model.saveModel(model.forest, features);

    cout << "dataset_test end line" << endl;
}

int main(){

//    dataset_test();
    lchf_model model;
    model.path = prefix;
    model.forest = model.loadForest();
    auto features = model.loadFeatures();
    google::protobuf::ShutdownProtobufLibrary();
    cout << "end" << endl;
    return 0;
}

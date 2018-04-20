#include "lchf.h"
#include "forest.h"
#include <memory>
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

static std::string prefix = "/home/meiqua/6DPose/cxxLCHF/test";

void dataset_test(){
    lchf_helper::Timer_lchf time;
    string pre = "/home/meiqua/6DPose/public/datasets/hinterstoisser/train/09/";
int count = 0;
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
            Linemod_feature f(rgb(aBox), depth(aBox));

//            Linemod_feature f(rgb, depth);
            if(f.constructEmbedding()){
                f.constructResponse();
                features.push_back(move(f));
            }
        }
//        cout << "features " << i << " OK" << endl;
//        cout << endl;
    }
    time.out("construct features");

    lchf_model model;
    vector<Info> infos;
    model.train(features, infos);
    time.out("train time:");

    model.path = prefix;
    model.saveModel(model.forest, features);

    cout << "dataset_test end line" << endl;
}

void fake_feature_test() {
    struct fake_feature{
      float x;
      float y;
      float similarity(const fake_feature& other) const {
          float dis = (x-other.x)*(x-other.x) + (y-other.y)*(y-other.y);
          return (100/(dis+1));
      }
    };
    vector<float> seed_center;
    for(int i=0;i<100;i++){
        seed_center.push_back(i*10);
    }
    vector<fake_feature> fs;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0,1);
    for(auto center: seed_center){
        for(int i=0;i<10;i++){
            auto number = distribution(generator);
//            number = 0;  //inner dis = 0; outer class dis > 100
            fake_feature f;
            f.x = center + number;
            f.y = center + number;
            fs.push_back(f);
        }
    }

    Forest<fake_feature> forest;
    vector<Info> infos;
    forest.Train(fs, infos);

    auto tree = forest.trees[0];
    for(int i=0;i<tree.id_leafnodes_.size();i++){
        auto leaf = tree.nodes_[tree.id_leafnodes_[i]];
        cout << i << "th leaf node "<<tree.id_leafnodes_[i]<<": " << endl;
        cout << "leaf node depth: " << leaf.depth << endl;
        cout << "parent node "<< leaf.pnode << endl;
        cout << "simi thresh: " <<
                tree.nodes_[leaf.pnode].simi_thresh <<endl;
        for(auto idx: leaf.ind_feats){
             cout << idx << endl;
        }
        cout << endl;
    }

    cout << "fake feature test end" << endl;
}

int main(){
//    dataset_test();

    fake_feature_test();

//    lchf_model model;
//    model.path = prefix;
//    model.forest = model.loadForest();
//    auto features = model.loadFeatures();
//    google::protobuf::ShutdownProtobufLibrary();
    cout << "end" << endl;
    return 0;
}

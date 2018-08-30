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

    int train_size = 1000;
    vector<Linemod_feature> features;
    vector<Info> infos;

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

    std::cout << "sample size: " << features.size() << std::endl;
    auto forest = lchf_model::train(features, infos);
    time.out("train time:");

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
      string name = "linemod";
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

void API_test(){

//    auto infos = lchf_model::loadInfos("/home/meiqua/6DPose/public/datasets/hinterstoisser/LCHF");
//    auto feats = lchf_model::loadFeatures("/home/meiqua/6DPose/public/datasets/hinterstoisser/LCHF");
//    auto forest = lchf_model::train(feats, infos);

    auto forest = lchf_model::loadForest("/home/meiqua/6DPose/public/datasets/hinterstoisser/LCHF");
    auto feats = lchf_model::loadFeatures("/home/meiqua/6DPose/public/datasets/hinterstoisser/LCHF");

    Mat rgb = imread("/home/meiqua/6DPose/cxxLCHF/test/0000_rgb.png");
    Mat depth = imread("/home/meiqua/6DPose/cxxLCHF/test/0000_dep.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

    int rows = depth.rows;
    int cols = depth.cols;
    int stride = 3;
    int width = 100;
    int height = 100;
    int dep_x = 10;
    int dep_y = 10;

    std::vector<std::vector<int>> rois;
    for(int x=0; x<cols-width-2*stride; x+=stride){
        for(int y=0; y<rows-height-2*stride; y+=stride){

            int dep_value = depth.at<ushort>(y+dep_y, x+dep_x);
            if(dep_value==0) continue;
            std::vector<int> roi = {x, y, width, height, dep_value};
            rois.push_back(roi);
        }
    }
    auto scene_feats = lchf_model::get_feats_from_scene(rgb, depth, rois);
    auto leaf_of_trees_of_scene = lchf_model::predict(forest, feats, scene_feats);
    auto leaf_feats_map = lchf_model::getLeaf_feats_map(forest);

    std::map<int, double> bg_prob;
    for(int scene_iter=0; scene_iter<leaf_of_trees_of_scene.size(); scene_iter++){
        auto& trees_of_scene = leaf_of_trees_of_scene[scene_iter];
        auto& roi = rois[scene_iter];
        for(int tree_iter=0; tree_iter<trees_of_scene.size(); tree_iter++){
            auto& leaf_iter = trees_of_scene[tree_iter];
            auto& leaf_map = leaf_feats_map[tree_iter];
            auto& predicted_ids = leaf_map[leaf_iter];

            for(auto id: predicted_ids){
                if(bg_prob.find(id) == bg_prob.end()){
                    bg_prob[id] = 1.0/predicted_ids.size()
                            /trees_of_scene.size()/leaf_of_trees_of_scene.size();
                }else{
                    bg_prob[id] += 1.0/predicted_ids.size()
                            /trees_of_scene.size()/leaf_of_trees_of_scene.size();
                }
            }

        }
    }
}

void simi_test(){
    string pre = "/home/meiqua/6DPose/public/datasets/hinterstoisser/train/06/";

    vector<Linemod_feature> features;
    int train_size = 1000;
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

        Linemod_feature f(rgb(bbox), depth(bbox));
        if(f.constructEmbedding()){
            f.constructResponse();
            features.push_back(f);
            cv::imshow("rgb", rgb);

//            {
//                float simi = features[features.size()-1]
//                                        .similarity(features[features.size()-1]);
//                std::cout << "\nself simi(should be 100): " << simi << std::endl;
//            }

//            { // result is around 75, 55-95
//                cv::Mat rgb_2, depth_2;
//                pyrDown(rgb(bbox), rgb_2);

//                imshow("rgb_2", rgb_2);

//                pyrDown(depth(bbox), depth_2);
//                depth_2 *= 2;

//                Linemod_feature f_2(rgb_2, depth_2);
//                f_2.constructResponse();
//                float simi = features[features.size()-1]
//                                        .similarity(f_2);
//                std::cout << "\ndiff depth simi: " << simi << std::endl;
//            }

//            if(features.size() > 1){
//                float simi = features[features.size()-2]
//                        .similarity(features[features.size()-1]);
//                std::cout << "adj simi: " << simi << std::endl;
//            }
        }

    }
}
int main(){
//    API_test();
    simi_test();
    //    google::protobuf::ShutdownProtobufLibrary();
    cout << "end" << endl;
    return 0;
}

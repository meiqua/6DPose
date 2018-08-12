#include "forest.h"
#include<memory>
#include <functional>
#include <random>
#include <math.h>
#include <fstream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

// lchf_model
Forest<Linemod_feature> lchf_model::train(const std::vector<Linemod_feature> &feats, const std::vector<Info>& infos)
{
    Forest<Linemod_feature> forest;
    forest.Train(feats, infos);
    return forest;
}
std::vector<std::vector<int>> lchf_model::predict(const Forest<Linemod_feature>& forest, const std::vector<Linemod_feature> &templ_feats, const std::vector<Linemod_feature> &scene_feats)
{
    std::vector<std::vector<int>> scene_leafs;
    for(auto& f: scene_feats){
        auto leafs = forest.Predict(templ_feats, f);
        scene_leafs.push_back(leafs);
    }
    return scene_leafs;
}

void lchf_model::saveForest(Forest<Linemod_feature> &forest, std::string path){
    fs::path dir(path);
    fs::path file("forests");
    auto full_path = dir / file;
    fstream output(full_path.c_str(), ios::out | ios::trunc | ios::binary);
    lchf::Forest forest_;
    forest.write(&forest_);
    if(!forest_.SerializeToOstream(&output)){
        cerr << "Fail to write forests" << endl;
    }
}
Forest<Linemod_feature> lchf_model::loadForest(std::string path)
{
    fs::path file("forests");
    lchf::Forest forest_;

    fs::path dir(path);
    auto full_path = dir / file;
    fstream input(full_path, ios::in | ios::binary);
    if(!forest_.ParseFromIstream(&input)){
        cerr << "Fail to read forest" << endl;
        return -1;
    }
    Forest<Linemod_feature> forest;
    forest.read(forest_);
    return forest;
}

void lchf_model::saveFeatures(std::vector<Linemod_feature> &features, std::string path){
    fs::path dir(path);
    if(!features.empty())
    {
        fs::path file("features");
        auto full_path = dir / file;
        fstream output(full_path.c_str(), ios::out | ios::trunc | ios::binary);
        lchf::Linemod_features fs;
        for(auto& feature: features){
            auto fs_ = fs.add_features();
            feature.write(fs_);
        }
        if(!fs.SerializeToOstream(&output)){
            cerr << "Fail to write features" << endl;
        }
    }
}
std::vector<Linemod_feature> lchf_model::loadFeatures(std::string path)
{
    fs::path file("features");
    lchf::Linemod_features features;
    fs::path dir(path);
    auto full_path = dir / file;
    fstream input(full_path, ios::in | ios::binary);
    if(!features.ParseFromIstream(&input)){
        cerr << "Fail to read forest" << endl;
        return std::vector<Linemod_feature>();
    }
    int feats_size = features.features_size();
    std::vector<Linemod_feature> feats(feats_size);
    for(int i=0;i<feats_size;i++){
//        cout << feats_size<< " features, " << "reading feature " << i << endl;
        feats[i].read(features.features(i));
    }
    return feats;
}

void lchf_model::saveInfos(std::vector<Info> &cluster_infos, std::string path)
{
    fs::path dir(path);
    {
        fs::path file("infos");
        auto full_path = dir / file;
        fstream output(full_path.c_str(), ios::out | ios::trunc | ios::binary);
        lchf::Infos infos;
        for(auto& info: cluster_infos){
            auto info_ = infos.add_info();
            info.write(info_);
        }
        if(!infos.SerializeToOstream(&output)){
            cerr << "Fail to write cluster infos" << endl;
        }
    }
}
std::vector<Info> lchf_model::loadInfos(std::string path)
{
    fs::path file("infos");
    lchf::Infos infos;
    fs::path dir(path);
    auto full_path = dir / file;
    fstream input(full_path, ios::in | ios::binary);
    if(!infos.ParseFromIstream(&input)){
        cerr << "Fail to read cluster infos" << endl;
        return std::vector<Info>();
    }
    int infos_size = infos.info_size();
    std::vector<Info> infos_(infos_size);
    for(int i=0;i<infos_size;i++){
        infos_[i].read(infos.info(i));
    }
    return infos_;
}

// lchf_helper
float lchf_helper::getMean(std::vector<float>& v){
    float sum = std::accumulate(v.begin(), v.end(), 0.0);
    float  mean = sum / v.size();
    return mean;
}
float lchf_helper::getDev(std::vector<float>& v){
    float mean = getMean(v);
    std::vector<float> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](float x) { return x - mean; });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdev = std::sqrt(sq_sum / v.size());
    return stdev;
}

bool lchf_helper::isRotationMatrix(cv::Mat &R){
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  norm(I, shouldBeIdentity) < 1e-6;
}
template<class type>
cv::Vec3f lchf_helper::rotationMatrixToEulerAngles(cv::Mat &R){
    assert(lchf_helper::isRotationMatrix(R));
    float sy = std::sqrt(R.at<type>(0,0) * R.at<type>(0,0) +  R.at<type>(1,0) * R.at<type>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = std::atan2(R.at<type>(2,1) , R.at<type>(2,2));
        y = std::atan2(-R.at<type>(2,0), sy);
        z = std::atan2(R.at<type>(1,0), R.at<type>(0,0));
    }
    else
    {
        x = std::atan2(-R.at<type>(1,2), R.at<type>(1,1));
        y = std::atan2(-R.at<type>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}
template<class type>
cv::Mat lchf_helper::eulerAnglesToRotationMatrix(cv::Vec3f &theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<type>(3,3) <<
               1,       0,              0,
               0,       std::cos(theta[0]),   -std::sin(theta[0]),
               0,       std::sin(theta[0]),   std::cos(theta[0])
               );
    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<type>(3,3) <<
               std::cos(theta[1]),    0,      std::sin(theta[1]),
               0,               1,      0,
               -std::sin(theta[1]),   0,      std::cos(theta[1])
               );
    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<type>(3,3) <<
               std::cos(theta[2]),    -std::sin(theta[2]),      0,
               std::sin(theta[2]),    std::cos(theta[2]),       0,
               0,               0,                  1);
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;
    return R;
}

void lchf_helper::cluster(std::vector<Info> &input, std::vector<Info> &output){
    std::vector<std::vector<double>> points_in;
    for(auto& info: input){
        auto& r_v = info.rpy;  //R is float
        std::vector<double> point_in;
        for(int i=0;i<3;i++){
            point_in.push_back(r_v.at<float>(i,0));
            point_in.push_back(info.t.at<float>(i,0));
        }
        points_in.push_back(point_in);
    }
    auto msp = MeanShift();
    auto clusters = msp.cluster(points_in, 1);  //second param is sigma of gussian
    for(auto& clu: clusters){
        auto& mode = clu.mode;
        cv::Vec3f r_v;
        cv::Mat t = cv::Mat(3,1,CV_32FC1);
        for(int i=0;i<3;i++){
            r_v[i] = mode[i];
            t.at<float>(i,0) = mode[i+3];
        }

        Info info;
        info.rpy = r_v;
        info.t = t;
        info.id = input[0].id;
        output.push_back(std::move(info));
    }
}
template <typename T>
std::vector<size_t> lchf_helper::sort_indexes(const std::vector<T> &v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  return idx;
}

std::vector<std::map<int, std::vector<int>>> lchf_model::getLeaf_feats_map(const Forest<Linemod_feature> &forest)
{
    std::vector<std::map<int, std::vector<int>>> forest_leaf_map(forest.trees.size());
    for(int i=0; i<forest.trees.size(); i++){
        auto& tree = forest.trees[i];
        auto& leaf_map = forest_leaf_map[i];
        for(auto leaf: tree.id_leafnodes_){
            leaf_map[leaf] = tree.nodes_[leaf].ind_feats;
        }
    }
    return forest_leaf_map;
}

std::vector<Linemod_feature> lchf_model::get_feats_from_scene(Mat &rgb, Mat &depth, std::vector<std::vector<int> > &rois)
{
    assert(rois.size()>0);
    assert(rois[0].size() == 5);

    Linemod_feature whole_scene(rgb, depth);
    whole_scene.constructResponse();

    std::vector<Linemod_feature> feats;
    for(auto& roi: rois){
        Linemod_embedding embedding;
        embedding.center_dep = roi[5];
        {
            std::vector<cv::Mat> rgb_response;
            for(cv::Mat& res: whole_scene.embedding.rgb_response){
                cv::Rect crop(roi[0], roi[1], roi[2], roi[3]);
                cv::Mat res_roi = res(crop).clone();
                rgb_response.push_back(res_roi);
            }
            embedding.rgb_response = rgb_response;
        }
        {
            std::vector<cv::Mat> dep_response;
            for(cv::Mat& res: whole_scene.embedding.dep_response){
                cv::Rect crop(roi[0], roi[1], roi[2], roi[3]);
                cv::Mat res_roi = res(crop).clone();
                dep_response.push_back(res_roi);
            }
            embedding.dep_response = dep_response;
        }
        Linemod_feature feat;
        feat.depth = depth(cv::Rect(roi[0], roi[1], roi[2], roi[3]));
        feat.setEmbedding(embedding);
        feats.push_back(feat);
    }
    return feats;
}

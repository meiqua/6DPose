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

void Info_cluster::cluster(std::vector<Info> &input, std::vector<Info> &output){
    vector<vector<double>> points_in;
    for(auto& info: input){
        auto& R = info.R;  //R is float
        auto r_v = rotationMatrixToEulerAngles<float>(R);
        vector<double> point_in;
        for(int i=0;i<3;i++){
            point_in.push_back(r_v[i]);
            point_in.push_back(info.t.at<float>(i,0));
        }
        points_in.push_back(point_in);
    }
    auto msp = MeanShift();
    auto clusters = msp.cluster(points_in, 1);  //second param is sigma of gussian
    for(auto& clu: clusters){
        auto& mode = clu.mode;
        cv::Vec3f r_v;
        Mat t = Mat(3,1,CV_32FC1);
        for(int i=0;i<3;i++){
            r_v[i] = mode[i];
            t.at<float>(i,0) = mode[i+3];
        }
        auto R = eulerAnglesToRotationMatrix<float>(r_v);
        Info info;
        info.R = R;
        info.t = t;
        info.id = input[0].id;
        output.push_back(move(info));
    }
}

void lchf_model::train(const std::vector<Linemod_feature> &feats)
{
    forest.Train(feats);
}

std::vector<int> lchf_model::predict(const std::vector<Linemod_feature> &feats, Linemod_feature &f)
{
    return forest.Predict(feats, f);
}

Forest<Linemod_feature> lchf_model::loadForest()
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

std::vector<Linemod_feature> lchf_model::loadFeatures()
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

std::vector<Info> lchf_model::loadCluster_infos()
{
    fs::path file("cluster_infos");
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

void lchf_model::saveModel(Forest<Linemod_feature> &forest, std::vector<Linemod_feature> &features){
    fs::path dir(path);
    if(!features.empty()){
        fs::path file("features");
        auto full_path = dir / file;
        fstream output(full_path.c_str(), ios::out | ios::trunc | ios::binary);
        lchf::Linemod_features fs;
        if(params.save_src){
            for(auto& feature: features){
                auto fs_ = fs.add_features();
                feature.write(fs_, 1, 1);
            }
        }else if(params.save_all_embedding){
            for(auto& feature: features){
                auto fs_ = fs.add_features();
                feature.write(fs_, 0, 1);
            }
        }else if(params.save_split_embedding){
            vector<int> check_array(features.size(), 0);
            for(auto& tree: forest.trees){
                for(auto& idx: tree.id_non_leafnodes_){
                    check_array[idx] = 1;
                }
            }
            for(int i=0;i<check_array.size();i++){
                auto fs_ = fs.add_features();
                if(check_array[i]){
                    features[i].write(fs_, 0, 1);
                }
            }
        }
        if(!fs.SerializeToOstream(&output)){
            cerr << "Fail to write features" << endl;
        }
    }
    {
        fs::path file("forests");
        auto full_path = dir / file;
        fstream output(full_path.c_str(), ios::out | ios::trunc | ios::binary);
        lchf::Forest forest_;
        forest.write(&forest_);
        if(!forest_.SerializeToOstream(&output)){
            cerr << "Fail to write forests" << endl;
        }
    }
}

void lchf_model::saveInfos(std::vector<Info> &cluster_infos)
{
    fs::path dir(path);
    {
        fs::path file("cluster_infos");
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

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

template<class Feature>
void Tree<Feature>::train(const std::vector<Feature> &feats,
                                const std::vector<int>& index){
    //root
    nodes_[0].issplit = false;
    nodes_[0].pnode = 0;
    nodes_[0].depth = 1;
    nodes_[0].cnodes[0] = 0;
    nodes_[0].cnodes[1] = 0;
    nodes_[0].isleafnode = 1;
    nodes_[0].ind_feats = index;


    num_nodes_ = 1;
    num_leafnodes_ = 1;

    bool stop = 0;
    int num_nodes = 1;
    int num_leafnodes = 1;
    vector<int> lcind,rcind;
    int num_nodes_iter;
    int num_split;
    while(!stop){ // restart when we finish spliting old nodes
        num_nodes_iter = num_nodes_;
        num_split = 0;
        for (int n = 0; n < num_nodes_iter; n++ ){
            if (!nodes_[n].issplit){
                if (nodes_[n].depth == max_depth_ ||
                        nodes_[n].ind_feats.size()<size_thresh_){
                    nodes_[n].issplit = true;
                }else{
                    Split(feats, nodes_[n].ind_feats, nodes_[n].split_feat_idx, lcind, rcind);

                    nodes_[n].issplit = true;
                    nodes_[n].isleafnode = false;
                    nodes_[n].cnodes[0] = num_nodes ;
                    nodes_[n].cnodes[1] = num_nodes +1;

                    //add left and right child nodes into the random tree
                    nodes_[num_nodes].ind_feats = lcind;
                    nodes_[num_nodes].issplit = false;
                    nodes_[num_nodes].pnode = n;
                    nodes_[num_nodes].depth = nodes_[n].depth + 1;
                    nodes_[num_nodes].cnodes[0] = 0;
                    nodes_[num_nodes].cnodes[1] = 0;
                    nodes_[num_nodes].isleafnode = true;

                    nodes_[num_nodes +1].ind_feats = rcind;
                    nodes_[num_nodes +1].issplit = false;
                    nodes_[num_nodes +1].pnode = n;
                    nodes_[num_nodes +1].depth = nodes_[n].depth + 1;
                    nodes_[num_nodes +1].cnodes[0] = 0;
                    nodes_[num_nodes +1].cnodes[1] = 0;
                    nodes_[num_nodes +1].isleafnode = true;

                    num_split++;
                    num_leafnodes++;
                    num_nodes +=2;
                }
            }
        }
        if (num_split == 0){  // no new node to split
            stop = 1;
        }
        else{
            num_nodes_ = num_nodes;
            num_leafnodes_ = num_leafnodes;
        }
    }
    id_leafnodes_.clear();
    id_non_leafnodes_.clear();
    for (int i=0;i < num_nodes_;i++){
        if (nodes_[i].isleafnode == 1){
            id_leafnodes_.push_back(i);
        }else {
            id_non_leafnodes_.push_back(i);
        }
    }
}

template<class Feature>
void Tree<Feature>::Split(const std::vector<Feature> &feats, const std::vector<int>& ind_feats,
                           int f_idx, std::vector<int> &lcind, std::vector<int> &rcind)
{
    if(ind_feats.size()==0){
        f_idx = 0;
        lcind.clear();
        rcind.clear();
        return;
    }
    std::random_device rd;
    unsigned long seed = rd();
    std::mt19937 engine(seed);
    std::vector<int> distribution(ind_feats.size(),1);

    int attempts = std::min(split_attempts_, int(ind_feats.size()));
    float max_info_gain = 0; int best_feat = 0;
    std::vector<int> lcind_best, rcind_best;
    for(int attempt=0; attempt < attempts; attempt++){
        std::discrete_distribution<> dist(distribution.begin(), distribution.end());
        auto rng = std::bind(dist, std::ref(engine));
        int select = rng();

        std::vector<int> lcind_local, rcind_local;
        int left = 0; int right = 0;
        for(int idx=0; idx<ind_feats.size(); idx++){
            if(select == idx){
                continue;
            }else{
                float simi = feats[ind_feats[select]].similarity(feats[ind_feats[idx]]);
                if(simi <= simi_thresh_){
                    left++;
                    lcind_local.push_back(idx);
                }else if(simi > simi_thresh_){
                    right++;
                    rcind_local.push_back(idx);
                }
            }
        }

        float pro = float(left)/(left+right);
        float info_gain = pro*std::log2f(pro)+(1-pro)*std::log2f((1-pro));
        if(info_gain>max_info_gain){
            max_info_gain = info_gain;
            best_feat = select;
            lcind_best = lcind_local;
            rcind_best = rcind_local;
        }
        // we don't want any more~~
        distribution[select] = 0;
    }

    lcind.clear();
    rcind.clear();
    lcind = lcind_best;
    rcind = rcind_best;

    f_idx = ind_feats[best_feat];
}

template<class Feature>
int Tree<Feature>::predict(Feature &f)
{
    auto& current = nodes_[0];
    int current_idx = 0;
    while(!current.isleafnode){
        if(f.similarity(nodes_[current.split_feat_idx]) <= simi_thresh_){
            current_idx = current.cnodes[0];
            current = nodes_[current_idx];
        }else{
            current_idx = current.cnodes[1];
            current = nodes_[current_idx];
        }
    }
    return current_idx;
}

template<class Feature>
void Forest<Feature>::Train(const std::vector<Feature> &feats)
{
    std::random_device rd;
    unsigned long seed = rd();
    std::mt19937 engine(seed);
    std::vector<int> distribution(feats.size(),1);

    size_t train_size = size_t(feats.size()*train_ratio_);

    for(auto& tree: trees){
        vector<int> ind_feats(train_size);
        for(size_t i=0; i<train_size; i++){
            std::discrete_distribution<> dist(distribution.begin(), distribution.end());
            auto rng = std::bind(dist, std::ref(engine));
            int select = rng();
            ind_feats[i] = select;
            distribution[select] = 0;
        }
        tree.train(feats, ind_feats);
    }
}

template<class Feature>
std::vector<int> Forest<Feature>::Predict(Feature &f)
{
    vector<int> results;
    for(auto& tree: trees){
        auto result = tree.predict(f);
        results.push_back(result);
    }
    return results;
}

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

void lchf_model::saveModel(Forest<Linemod_feature> &forest, std::vector<Linemod_feature> &features,
                           std::vector<Info> &cluster_infos){
    fs::path dir(path);
    if(params.save_leaf_distribution&&(!cluster_infos.empty())){
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
    if(!features.empty()){
        fs::path file("features");
        auto full_path = dir / file;
        fstream output(full_path.c_str(), ios::out | ios::trunc | ios::binary);
        lchf::Linemod_features fs;
        if(params.save_src){
            for(auto& feature: features){
                auto fs_ = fs.add_features();
                feature.write(fs_, 1, 1, 1);
            }
        }else if(params.save_all_embedding&&params.save_all_info){
            for(auto& feature: features){
                auto fs_ = fs.add_features();
                feature.write(fs_, 0, 1, 1);
            }
        }else if(params.save_split_embedding&&params.save_leaf_info){
            vector<int> check_array(features.size(), 0);
            for(auto& tree: forest.trees){
                for(auto& idx: tree.id_non_leafnodes_){
                    check_array[idx] = 1;
                }
            }
            for(int i=0;i<check_array.size();i++){
                auto fs_ = fs.add_features();
                if(check_array[i]){
                    features[i].write(fs_, 0, 1, 1);
                }else{
                    features[i].write(fs_, 0, 0, 1);
                }
            }
        }else if(params.save_split_embedding&&params.save_leaf_distribution){
            vector<int> check_array(features.size(), 0);
            for(auto& tree: forest.trees){
                for(auto& idx: tree.id_non_leafnodes_){
                    check_array[idx] = 1;
                }
            }
            for(int i=0;i<check_array.size();i++){
                auto fs_ = fs.add_features();
                if(check_array[i]){
                    features[i].write(fs_, 0, 1, 1);
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

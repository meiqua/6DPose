#include "forest.h"
#include<memory>
#include <functional>
#include <random>
#include <math.h>
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
    for (int i=0;i < num_nodes_;i++){
        if (nodes_[i].isleafnode == 1){
            id_leafnodes_.push_back(i);
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

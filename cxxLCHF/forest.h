#ifndef FOREST_H
#define FOREST_H
#include "meanshift/MeanShift.h"
#include "lchf.h"
#include <numeric>
#include <chrono>
#include <cmath>
namespace lchf_helper {
float getMean(std::vector<float>& v);
float getDev(std::vector<float>& v);
bool isRotationMatrix(cv::Mat &R);
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v);
template<class type>
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);
template<class type>
cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta);
void cluster(std::vector<Info> &input, std::vector<Info> &output);
class Timer_lchf
{
public:
    Timer_lchf() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << ", elasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
}

class Node {
public:
    bool issplit=0;
    int pnode=0;
    int depth=0;
    int cnodes[2]={0};
    bool isleafnode=0;
    int split_feat_idx=0;
    float simi_thresh=50;
    std::vector<int> ind_feats;

   void write(lchf::Node* node){
       node->set_simi_thresh(simi_thresh);
        node->set_issplit(issplit);
        node->set_pnode(pnode);
        node->add_cnodes(cnodes[0]);
        node->add_cnodes(cnodes[1]);
        node->set_depth(depth);
        node->set_isleafnode(isleafnode);
        node->set_split_feat_idx(split_feat_idx);
        for(auto idx: ind_feats){
            node->add_ind_feats(idx);
        }
    }
    void read(const lchf::Node& node){
        simi_thresh = node.simi_thresh();
        issplit = node.issplit();
        pnode = node.pnode();
        cnodes[0] = node.cnodes(0);
        cnodes[1] = node.cnodes(1);
        depth = node.depth();
        isleafnode = node.isleafnode();
        split_feat_idx = node.split_feat_idx();
        int ind_num = node.ind_feats_size();
        ind_feats.resize(ind_num);
        for(int i=0;i<ind_num;i++){
            ind_feats[i] = node.ind_feats(i);
        }
    }
};

template <class Feature>
class Tree {
public:
    int max_depth_;
    int num_leafnodes_;
    int num_nodes_;
    int size_thresh_;
    int split_attempts_;
    float simi_thresh_;

    std::vector<int> id_leafnodes_;
    std::vector<int> id_non_leafnodes_;
    std::vector<Node> nodes_;

    void write(lchf::Tree* tree){
        tree->set_max_depth_(max_depth_);
        tree->set_num_leafnodes_(num_leafnodes_);
        tree->set_num_nodes_(num_nodes_);
        tree->set_size_thresh_(size_thresh_);
        tree->set_split_attempts_(split_attempts_);
        tree->set_simi_thresh_(simi_thresh_);
        for(auto idx: id_leafnodes_){
            tree->add_id_leafnodes_(idx);
        }
        for(auto idx: id_non_leafnodes_){
            tree->add_id_non_leafnodes_(idx);
        }
        for(auto& node: nodes_){
            auto node_ = tree->add_nodes_();
            node.write(node_);
        }
    }
    void read(const lchf::Tree& tree){
        max_depth_ = tree.max_depth_();
        num_leafnodes_ = tree.num_leafnodes_();
        num_nodes_ = tree.num_nodes_();
        size_thresh_ = tree.size_thresh_();
        split_attempts_ = tree.split_attempts_();
        simi_thresh_ = tree.simi_thresh_();
        int id_leafnodes_size = tree.id_leafnodes__size();
        id_leafnodes_.resize(id_leafnodes_size);
        for(int i=0;i<id_leafnodes_size;i++){
            id_leafnodes_[i] = tree.id_leafnodes_(i);
        }
        int id_non_size = tree.id_non_leafnodes__size();
        id_non_leafnodes_.resize(id_non_size);
        for(int i=0;i<id_non_size;i++){
            id_non_leafnodes_[i] = tree.id_non_leafnodes_(i);
        }
        int node_size = tree.nodes__size();
        nodes_.resize(node_size);
        for(int i=0;i<node_size;i++){
            nodes_[i].read(tree.nodes_(i));
        }
    }

    Tree(float simi_thresh=50, int max_depth=32, int size_thresh=10, int split_attempts=128){
        size_thresh_ = size_thresh;
        max_depth_ = max_depth;
        split_attempts_=split_attempts;
        simi_thresh_ = simi_thresh;
    }

    void train(const std::vector<Feature>& feats, const std::vector<Info>& infos
               ,const std::vector<int>& index);
    float info_gain(const std::vector<Info>& infos, const std::vector<int>& ind_feats,
                    const std::vector<int>& left,
                    const std::vector<int>& right, const std::vector<float>& simis, int depth);


    bool split_linemod(const std::vector<Feature>& feats, const std::vector<Info>& infos
               , const std::vector<int>& ind_feats,
               int& f_idx, std::vector<int>& lcind, std::vector<int>& rcind, float& simi_thresh, int depth);
    int predict_linemod(const std::vector<Feature> &feats, const Feature& f) const;

    template <typename ...Params>
    bool split(std::string name, Params&&... params){
        if(name == "linemod"){
            return split_linemod(std::forward<Params>(params)...);
        }else if(name == "cnn"){
            CV_Error(cv::Error::StsBadArg, "not yet");
        }
        else{
            CV_Error(cv::Error::StsBadArg, "unsupported feature type");
        }
    }

    template <typename ...Params>
    int predict(std::string name, Params&&... params) const{
        if(name == "linemod"){
            return predict_linemod(std::forward<Params>(params)...);
        }else if(name == "cnn"){
            CV_Error(cv::Error::StsBadArg, "not yet");
        }else{
            CV_Error(cv::Error::StsBadArg, "unsupported feature type");
        }
    }
};

template <class Feature>
class Forest {
public:
  std::vector<Tree<Feature> > trees;
  int max_numtrees_;
  double train_ratio_;
  Forest(int max_numtrees=5, double train_ratio = 0.8){
      max_numtrees_ = max_numtrees;
      trees.resize(max_numtrees_);
      train_ratio_ = train_ratio;
  }
  void Train(const std::vector<Feature>& feats, const std::vector<Info>& infos);
  std::vector<int> Predict(const std::vector<Feature> &feats, const Feature &f) const;

  void write(lchf::Forest* forest){
      forest->set_max_numtrees(max_numtrees_);
      forest->set_train_ratio(train_ratio_);
      for(auto& tree: trees){
          auto tree_ = forest->add_trees();
          tree.write(tree_);
      }
  }
  void read(lchf::Forest& forest){
      max_numtrees_ = forest.max_numtrees();
      train_ratio_ = forest.train_ratio();
      int trees_size = forest.trees_size();
      trees.resize(trees_size);
      for(int i=0;i<trees_size;i++){
          trees[i].read(forest.trees(i));
      }
  }
};

template<class Feature>
void Tree<Feature>::train(const std::vector<Feature> &feats,const std::vector<Info>& infos,
                                const std::vector<int>& index){
    //root
    nodes_.resize(1);
    nodes_[0].issplit = false;
    nodes_[0].pnode = 0;
    nodes_[0].depth = 1;
    nodes_[0].cnodes[0] = 0;
    nodes_[0].cnodes[1] = 0;
    nodes_[0].isleafnode = 1;
    nodes_[0].ind_feats = index;

    num_nodes_ = 1;
    num_leafnodes_ = 1;

    bool stop = false;
    int num_nodes = 1;
    int num_leafnodes = 1;
    std::vector<int> lcind,rcind;
    int num_nodes_iter = 0;
    int num_split = 0;
    while(!stop){ // restart when we finish spliting old nodes
        num_nodes_iter = num_nodes_;
        nodes_.resize(num_nodes_*2+1);
        num_split = 0;
        for (int n = 0; n < num_nodes_iter; n++ ){
            if (!nodes_[n].issplit){
                if (nodes_[n].depth == max_depth_ ||
                        nodes_[n].ind_feats.size()<=size_thresh_){
                    nodes_[n].issplit = true;
                    nodes_[n].isleafnode = true;
                }else{
                    bool success = split(feats[0].name, feats, infos, nodes_[n].ind_feats,
                                         nodes_[n].split_feat_idx, lcind, rcind,
                                            nodes_[n].simi_thresh, nodes_[n].depth);
                    if(success){
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
                    }else{
                        nodes_[n].issplit = true;
                        nodes_[n].isleafnode = true;
                    }
                }
            }
        }
        if (num_split == 0){  // no new node to split
            stop = true;
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
//            if(nodes_[i].ind_feats.size()>size_thresh_)
//            std::cout << "leaf node "<< i<< " idx size: " << nodes_[i].ind_feats.size() << std::endl;
        }else {
            id_non_leafnodes_.push_back(i);
        }
    }
    nodes_.resize(num_nodes_);
}

template<class Feature>
bool Tree<Feature>::split_linemod(const std::vector<Feature> &feats, const std::vector<Info>& infos,
                          const std::vector<int>& ind_feats,
                           int& f_idx, std::vector<int> &lcind, std::vector<int> &rcind,
                           float& simi_thresh, int depth)
{
    if(ind_feats.size()==0){
        f_idx = 0;
        lcind.clear();
        rcind.clear();
        return false;
    }
    std::random_device rd;
    unsigned long seed = rd();
    std::mt19937 engine(seed);
    std::vector<int> distribution(ind_feats.size(),1);

    int attempts = std::min(split_attempts_, int(ind_feats.size()));

    float best_gain_all_feature = std::numeric_limits<float>::epsilon();
    float best_simis_all_feature = 0;
    int best_feature = 0;
    std::vector<int> left_best_all_feature, right_best_all_feature;
    for(int attempt=0; attempt < attempts; attempt++){
        std::discrete_distribution<> dist(distribution.begin(), distribution.end());
        auto rng = std::bind(dist, std::ref(engine));
        int select = rng();

        // calculate simis using selected one with others
        std::vector<float> simis(ind_feats.size(),0);
        for(int idx=0; idx<ind_feats.size(); idx++){
            if(select == idx){
                simis[idx] = -1;
            }else{
                auto& sel = feats[ind_feats[select]];
                auto& com = feats[ind_feats[idx]];
                float simi = sel.similarity(com);
                simis[idx] = simi;
            }
        }
//        for(auto simi: simis){
//            std::cout << int(simi) << "  ";
//        }
//        std::cout << std::endl;

        std::vector<int> dist_simis(simis.size(),0);
        //  sort idx by comparing simis
        auto sidxs = lchf_helper::sort_indexes(simis);
        int sample_count = 0;
        // we only want simi closer to simis center
        for(int i=simis.size()/4;i<simis.size()*3/4;i++){
            dist_simis[sidxs[i]] = 1;
            sample_count++;
        }
        int attempts2 = std::min(attempts, sample_count);

        std::vector<int> left_best, right_best;
        float best_gain = std::numeric_limits<float>::epsilon();// we want to have a positive gain
        float best_simi = 0;
        for(int i=0;i<attempts2;i++){
            std::vector<int> left, right;
            std::discrete_distribution<> dist_s(dist_simis.begin(), dist_simis.end());
            auto rng_simi = std::bind(dist_s, std::ref(engine));
            int sel_simi = rng_simi();

            // split to two sets using random sel simi
            for(int j=0;j<simis.size();j++){
                if(simis[j]>0){  // not self compare
                    if(simis[j]<=simis[sel_simi]){
                        left.push_back(j);
                    }else{
                        right.push_back(j);
                    }
                }
            }

            float gain = info_gain(infos, ind_feats, left, right, simis, depth);
            if(gain > best_gain){
                best_gain = gain;
                best_simi = simis[sel_simi];
                left_best = std::move(left);
                right_best = std::move(right);
            }
            dist_simis[sel_simi] = 0;
        }

        if(best_gain>best_gain_all_feature){
            best_gain_all_feature = best_gain;
            best_simis_all_feature = best_simi;
            best_feature = select;
            left_best_all_feature = std::move(left_best);
            right_best_all_feature = std::move(right_best);
        }
        distribution[select] = 0;
    }

    if(best_gain_all_feature>std::numeric_limits<float>::epsilon()*10){
        lcind.clear();
        lcind.reserve(left_best_all_feature.size());
        rcind.clear();
        rcind.reserve(right_best_all_feature.size());
        for(auto idx: left_best_all_feature){
            lcind.push_back(ind_feats[idx]);
        }
        for(auto idx: right_best_all_feature){
            rcind.push_back(ind_feats[idx]);
        }
        simi_thresh = best_simis_all_feature;
        f_idx = ind_feats[best_feature];
        return true;
    }else{
        return false;
    }
}

template<class Feature>
float Tree<Feature>::info_gain(const std::vector<Info>& infos,
                               const std::vector<int>& ind_feats,
                               const std::vector<int> &left,
                               const std::vector<int> &right, const std::vector<float> &simis, int depth)
{
    std::string type = "infos";
    if(type=="simis"){
        std::vector<float> left_simis, right_simis;
        for(auto idx: left){
            left_simis.push_back(simis[idx]);
        }
        for(auto idx: right){
            right_simis.push_back(simis[idx]);
        }

        if(left_simis.empty() || right_simis.empty() || ind_feats.empty())
            return 0;

        float left_w = float(left_simis.size())/(left_simis.size()+right_simis.size());
        float u1 = lchf_helper::getMean(left_simis);
        float u2 = lchf_helper::getMean(right_simis);
        // otsu method
        float var_reduce = left_w*(1-left_w)*(u1-u2)*(u1-u2);
        return var_reduce;
    }
    else if(type == "infos"){
        std::vector<int> left_infos, right_infos;
        for(auto idx: left){
            left_infos.push_back(ind_feats[idx]);
        }
        for(auto idx: right){
            right_infos.push_back(ind_feats[idx]);
        }

        if(left_infos.empty() || right_infos.empty() || ind_feats.empty())
            return 0;

        // calculate some metrics here, greater is better

        // refer to 3.2 info gain, rpy only
        // Real Time Head Pose Estimation with Random Regression Forests
        auto get_var = [&infos](std::vector<int> info_idxs){
            float mean[3] = {0};
            int num = info_idxs.size();
            cv::Mat A = cv::Mat(num, 3, CV_32FC1, cv::Scalar(0));
            for(int i=0; i<num; i++){
                const auto& info = infos[info_idxs[i]];
                for(int j=0; j<3; j++){
                    mean[j] += info.rpy.at<float>(j,0);
                    A.at<float>(i, j) = info.rpy.at<float>(j,0);
                }
            }
            {
                for(int j=0; j<3; j++){
                    mean[j] /= info_idxs.size();
                }
                cv::Mat mean_mat = cv::Mat(1, 3, CV_32FC1,  mean);
                cv::Mat ones = cv::Mat::ones(num, 1, CV_32FC1);

                A = A - ones*mean_mat;
            }

            cv::Mat var = A.t()*A/double(num);
            return var;
        };
        cv::Mat left_var = get_var(left_infos);
        cv::Mat right_var = get_var(right_infos);
        cv::Mat total_var = get_var(ind_feats);

        auto var_value = [](cv::Mat& var){return std::log2(cv::determinant(var));};
        float var_reduce = var_value(total_var) -
                (left_infos.size() *var_value(left_var)+
                 right_infos.size()*var_value(right_var))/infos.size();
        return var_reduce;
    }
    return 0;
}

template<class Feature>
int Tree<Feature>::predict_linemod(const std::vector<Feature> &feats, const Feature &f) const
{
    auto current = nodes_[0];
    int current_idx = 0;
    while(!current.isleafnode){
        if(feats[current.split_feat_idx].similarity(f) <= current.simi_thresh){
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
void Forest<Feature>::Train(const std::vector<Feature> &feats, const std::vector<Info>& infos)
{
    size_t train_size = size_t(feats.size()*train_ratio_);
    int count=0;
    lchf_helper::Timer_lchf time;
    for(auto& tree: trees){
        count++;
        std::cout << trees.size() << " trees, training tree: " << count << std::endl;
        std::vector<int> ind_feats(train_size);
        std::random_device rd;
        unsigned long seed = rd();
        std::mt19937 engine(seed);
        std::vector<int> distribution(feats.size(),1);
        for(size_t i=0; i<train_size; i++){
            std::discrete_distribution<> dist(distribution.begin(), distribution.end());
            auto rng = std::bind(dist, std::ref(engine));
            int select = rng();
            ind_feats[i] = select;
            distribution[select] = 0;
        }
        tree.train(feats, infos, ind_feats);
        time.out("training OK");
    }
}

template<class Feature>
std::vector<int> Forest<Feature>::Predict(const std::vector<Feature> &feats, const Feature &f) const
{
    std::vector<int> results;
    for(auto& tree: trees){
        auto result = tree.predict(feats[0].name, feats, f);
        results.push_back(result);
    }
    return results;
}

namespace lchf_model {
    Forest<Linemod_feature> train(const std::vector<Linemod_feature>& feats, const std::vector<Info>& infos);
    std::vector<std::vector<int>> predict(const Forest<Linemod_feature>& forest, const std::vector<Linemod_feature> &templ_feats, const std::vector<Linemod_feature> &scene_feats);

    std::vector<std::map<int, std::vector<int>>> getLeaf_feats_map(const Forest<Linemod_feature>& forest);

    std::vector<Linemod_feature> get_feats_from_scene(cv::Mat& rgb, cv::Mat& depth, std::vector<std::vector<int>>& rois);

    void saveForest(Forest<Linemod_feature>& forest, std::string path);
    Forest<Linemod_feature> loadForest(std::string path);

    void saveFeatures(std::vector<Linemod_feature>& features, std::string path);
    std::vector<Linemod_feature> loadFeatures(std::string path);

    void saveInfos(std::vector<Info>& infos, std::string path);
    std::vector<Info> loadInfos(std::string path);
};
#endif

#ifndef FOREST_H
#define FOREST_H
#include "meanshift/MeanShift.h"
#include "lchf.h"

class Node {
public:
    bool issplit=0;
    int pnode=0;
    int depth=0;
    int cnodes[2]={0};
    bool isleafnode=0;
    int split_feat_idx=0;
    std::vector<int> ind_feats;
    std::vector<int> ind_infos;

   void write(lchf::Node* node){
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
        for(auto idx: ind_infos){
            node->add_ind_infos(idx);
        }
    }
    void read(const lchf::Node& node){
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
        int ind_info_num = node.ind_infos_size();
        ind_infos.resize(ind_info_num);
        for(int i=0;i<ind_info_num;i++){
            ind_infos[i] = node.ind_infos(i);
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

    Tree(float simi_thresh=32, int max_depth=32, int size_thresh=16, int split_attempts=128){
        size_thresh_ = size_thresh;
        max_depth_ = max_depth;
        split_attempts_=split_attempts;
        simi_thresh_ = simi_thresh;
    }

    void train(const std::vector<Feature>& feats, const std::vector<int>& index);
    void Split(const std::vector<Feature>& feats, const std::vector<int>& ind_feats,
               int& f_idx, std::vector<int>& lcind, std::vector<int>& rcind);
    int predict(const std::vector<Feature> &feats, Feature& f);
};

template <class Feature>
class Forest {
public:
  std::vector<Tree<Feature> > trees;
  int max_numtrees_;
  double train_ratio_;
  Forest(int max_numtrees=10, double train_ratio = 0.8){
      max_numtrees_ = max_numtrees;
      trees.resize(max_numtrees_);
      train_ratio_ = train_ratio;
  }
  void Train(const std::vector<Feature>& feats);
  std::vector<int> Predict(const std::vector<Feature> &feats, Feature &f);

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
void Tree<Feature>::train(const std::vector<Feature> &feats,
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

    bool stop = 0;
    int num_nodes = 1;
    int num_leafnodes = 1;
    std::vector<int> lcind,rcind;
    int num_nodes_iter;
    int num_split;
    while(!stop){ // restart when we finish spliting old nodes
        num_nodes_iter = num_nodes_;
        nodes_.resize(num_nodes_*2+1);
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
//            if(nodes_[i].ind_feats.size()>size_thresh_)
//            std::cout << "leaf node "<< i<< " idx size: " << nodes_[i].ind_feats.size() << std::endl;
        }else {
            id_non_leafnodes_.push_back(i);
        }
    }
    nodes_.resize(num_nodes_);
}

template<class Feature>
void Tree<Feature>::Split(const std::vector<Feature> &feats, const std::vector<int>& ind_feats,
                           int& f_idx, std::vector<int> &lcind, std::vector<int> &rcind)
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
    float max_info_gain = -10000000; int best_feat = 0;
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
                auto& sel = feats[ind_feats[select]];
                auto& com = feats[ind_feats[idx]];
                float simi = sel.similarity(com);
//                std::cout << simi << std::endl;
                if(simi <= simi_thresh_){
                    left++;
                    lcind_local.push_back(ind_feats[idx]);
                }else if(simi > simi_thresh_){
                    right++;
                    rcind_local.push_back(ind_feats[idx]);
                }
            }
        }
        float sigma = 0.00001;  // avoid 0 or 1 for log
        float pro = float(left+sigma)/(left+right+sigma);
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
int Tree<Feature>::predict(const std::vector<Feature> &feats, Feature &f)
{
    auto& current = nodes_[0];
    int current_idx = 0;
    while(!current.isleafnode){
        if(f.similarity(feats[current.split_feat_idx]) <= simi_thresh_){
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
    int count=0;
    for(auto& tree: trees){
        count++;
        std::cout << trees.size() << " trees, training tree: " << count << std::endl;
        std::vector<int> ind_feats(train_size);
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
std::vector<int> Forest<Feature>::Predict(const std::vector<Feature> &feats, Feature &f)
{
    std::vector<int> results;
    for(auto& tree: trees){
        auto result = tree.predict(feats, f);
        results.push_back(result);
    }
    return results;
}


class Info_cluster {
public:
    void cluster(std::vector<Info>& input, std::vector<Info>& ouput);
private:
    bool isRotationMatrix(cv::Mat &R){
        cv::Mat Rt;
        transpose(R, Rt);
        cv::Mat shouldBeIdentity = Rt * R;
        cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
        return  norm(I, shouldBeIdentity) < 1e-6;
    }
    template<class type>
    cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R){
        assert(isRotationMatrix(R));
        float sy = sqrt(R.at<type>(0,0) * R.at<type>(0,0) +  R.at<type>(1,0) * R.at<type>(1,0) );

        bool singular = sy < 1e-6; // If

        float x, y, z;
        if (!singular)
        {
            x = atan2(R.at<type>(2,1) , R.at<type>(2,2));
            y = atan2(-R.at<type>(2,0), sy);
            z = atan2(R.at<type>(1,0), R.at<type>(0,0));
        }
        else
        {
            x = atan2(-R.at<type>(1,2), R.at<type>(1,1));
            y = atan2(-R.at<type>(2,0), sy);
            z = 0;
        }
        return cv::Vec3f(x, y, z);
    }
    template<class type>
    cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta)
    {
        // Calculate rotation about x axis
        cv::Mat R_x = (cv::Mat_<type>(3,3) <<
                   1,       0,              0,
                   0,       cos(theta[0]),   -sin(theta[0]),
                   0,       sin(theta[0]),   cos(theta[0])
                   );
        // Calculate rotation about y axis
        cv::Mat R_y = (cv::Mat_<type>(3,3) <<
                   cos(theta[1]),    0,      sin(theta[1]),
                   0,               1,      0,
                   -sin(theta[1]),   0,      cos(theta[1])
                   );
        // Calculate rotation about z axis
        cv::Mat R_z = (cv::Mat_<type>(3,3) <<
                   cos(theta[2]),    -sin(theta[2]),      0,
                   sin(theta[2]),    cos(theta[2]),       0,
                   0,               0,                  1);
        // Combined rotation matrix
        cv::Mat R = R_z * R_y * R_x;
        return R;
    }
};

class lchf_model {
public:
    Params params;
    std::string path;
    Forest<Linemod_feature> forest;
    void train(const std::vector<Linemod_feature>& feats);
    std::vector<int> predict(const std::vector<Linemod_feature> &feats, Linemod_feature &f);
    Forest<Linemod_feature> loadForest();
    std::vector<Linemod_feature> loadFeatures();
    std::vector<Info> loadCluster_infos();
    void saveModel(Forest<Linemod_feature>& forest, std::vector<Linemod_feature>& features);
    void saveInfos(std::vector<Info>& infos);
};
#endif

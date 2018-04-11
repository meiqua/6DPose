#ifndef FOREST_H
#define FOREST_H
#include "meanshift/MeanShift.h"
#include "lchf.h"

class Node {
public:
    bool issplit;
    int pnode;
    int depth;
    int cnodes[2];
    bool isleafnode;
    int split_feat_idx;
    std::vector<int> ind_feats;
    std::vector<int> ind_infos;
    Node(){
        ind_infos.clear();
        ind_feats.clear();
        issplit = 0;
        pnode = 0;
        depth = 0;
        cnodes[0] = 0;
        cnodes[1] = 0;
        isleafnode = 0;
    }
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
    int max_numnodes_;
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
        tree->set_max_numnodes_(max_numnodes_);
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
        max_numnodes_ = tree.max_numnodes_();
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

    Tree(float simi_thresh=50, int max_depth=20, int size_thresh=5, int split_attempts=100){
        size_thresh_ = size_thresh;
        max_depth_ = max_depth;
        max_numnodes_ = pow(2, max_depth_)-1;
        nodes_.resize(max_numnodes_);
        split_attempts_=split_attempts;
        simi_thresh_ = simi_thresh;
    }

    void train(const std::vector<Feature>& feats, const std::vector<int>& index);
    void Split(const std::vector<Feature>& feats, const std::vector<int>& ind_feats,
               int f_idx, std::vector<int>& lcind, std::vector<int>& rcind);
    int predict(Feature& f);
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
  std::vector<int> Predict(Feature &f);

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
    Forest<Linemod_feature> loadForest();
    std::vector<Linemod_feature> loadFeatures();
    std::vector<Info> loadCluster_infos();
    void saveModel(Forest<Linemod_feature>& forest, std::vector<Linemod_feature>& features,
                    std::vector<Info>& cluster_infos);
};
#endif

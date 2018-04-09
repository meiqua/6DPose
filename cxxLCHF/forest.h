#ifndef FOREST_H
#define FOREST_H

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

    Node(){
        ind_feats.clear();
        issplit = 0;
        pnode = 0;
        depth = 0;
        cnodes[0] = 0;
        cnodes[1] = 0;
        isleafnode = 0;
    }
    // read write
};

template <class Feature>
class Tree {
    int max_depth_;
    int max_numnodes_;
    int num_leafnodes_;
    int num_nodes_;
    int size_thresh_;
    int split_attempts_;
    float simi_thresh_;

    std::vector<int> id_leafnodes_;
    std::vector<Node> nodes_;

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
  std::vector<Tree<Feature> > trees;
  int max_numtrees_;
  double train_ratio_;
  Forest(int max_numtrees=10, double train_ratio = 0.8){
      max_numtrees_ = max_numtrees;
      trees.resize(max_numtrees_);
      train_ratio_ = train_ratio;
  }
  void Train(const std::vector<Feature>& feats);
};
#endif

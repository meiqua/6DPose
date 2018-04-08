#ifndef FOREST_H
#define FOREST_H

#include "lchf.h"

template <class ActualFeature>
class Node {
public:
    bool issplit;
    int pnode;
    int depth;
    int cnodes[2];
    bool isleafnode;
    double thresh;
    Feature<ActualFeature> split_feat;
    std::vector<int> ind_feats;

    //Constructors
    Node(){
        ind_feats.clear();
        issplit = 0;
        pnode = 0;
        depth = 0;
        cnodes[0] = 0;
        cnodes[1] = 0;
        isleafnode = 0;
        thresh = 0;
    }
    // read write
};

template <class ActualFeature>
class Tree {
    // depth of the tree:
    int max_depth_;
    // number of maximum nodes:
    int max_numnodes_;
    //number of leaf nodes and nodes
    int num_leafnodes_;
    int num_nodes_;

    // leafnodes id
    std::vector<int> id_leafnodes_;
    // tree nodes
    std::vector<Node<ActualFeature> > nodes_;

    Tree(int max_depth=20){
        max_depth_ = max_depth;
        max_numnodes_ = pow(2, max_depth_)-1;
        nodes_.resize(max_numnodes_);
    }

    void train(const std::vector<Feature<ActualFeature> >& feats, const std::vector<int>& index);
};

#endif

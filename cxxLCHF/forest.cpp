#include "forest.h"
using namespace std;
using namespace cv;

template<class ActualFeature>
void Tree<ActualFeature>::train(const std::vector<Feature<ActualFeature> > &feats,
                                const std::vector<int>& index){
    //root
    nodes_[0].issplit = false;
    nodes_[0].pnode = 0;
    nodes_[0].depth = 1;
    nodes_[0].cnodes[0] = 0;
    nodes_[0].cnodes[1] = 0;
    nodes_[0].isleafnode = 1;
    nodes_[0].thresh = 0;
    nodes_[0].ind_feats = index;

    bool stop = 0;
    num_nodes_ = 1;
    num_leafnodes_ = 1;
    int num_nodes = 1;
    int num_leafnodes = 1;
    float thresh;
    vector<int> lcind,rcind;
    int num_nodes_iter;
    int num_split;
    while(!stop){
        num_nodes_iter = num_nodes_;
        num_split = 0;
        for (int n = 0; n < num_nodes_iter; n++ ){
            if (!nodes_[n].issplit){
                if (nodes_[n].depth == max_depth_){
                    nodes_[n].issplit = true;
                }else{
                    //splitnode

                    nodes_[n].thresh  = thresh;
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
        if (num_split == 0){
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

#ifndef GLOBAL_H
#define GLOBAL_H

class Params {
public:
    int serialization_level = 0;
    bool save_src=0;
    bool save_all_embedding=0;
    bool save_split_embedding=0;
    bool save_all_info=0;
    bool save_leaf_info=0;
    bool save_leaf_distribution=0;
    Params(){
        if(serialization_level==0){
            save_src = 1;
            save_all_embedding = 1;
            save_all_info = 1;
        }else if(serialization_level==1){
            save_all_embedding=1;
            save_all_info=1;
        }else if(serialization_level==2){
            save_split_embedding = 1;
            save_leaf_info = 1;
        }else if (serialization_level==3) {
            save_split_embedding=1;
            save_leaf_distribution=1;
        }
    }
};

#endif

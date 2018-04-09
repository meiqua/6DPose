#ifndef LCHF_H
#define LCHF_H
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "proto/serialization.pb.h"
class Info {
public:
    cv::Mat R;
    cv::Mat t;
    std::string id;

    lchf::Info write();
    void read(lchf::Info &info_);
};

class Linemod_embedding {
public:
    Linemod_embedding():
        weak_threshold(10.0f),
        strong_threshold(55.0f),
        num_features(63),
        distance_threshold(2000),
        difference_threshold(50),
        extract_threshold(2),
        z_check(50){}
    float weak_threshold, strong_threshold;
    int num_features, distance_threshold, difference_threshold, extract_threshold;
    class element {
    public:
        element():x(0), y(0), label(0){}
        element(int x_, int y_, int label_): x(x_), y(y_), label(label_){}
        int x;
        int y;
        int label;

    };
    struct Candidate {
        Candidate(int x, int y, int label, float score);
        /// Sort candidates with high score to the front
        bool operator<(const Candidate& rhs) const
        {
          return score > rhs.score;
        }
        element f;
        float score;
    };

    std::vector<element> rgb_embedding;
    std::vector<element> depth_embedding;
    cv::Mat angle, normal;
    int center_dep;
    int z_check;

    lchf::Linemod_embedding write();
    void read(lchf::Linemod_embedding& embedding_);
};
inline Linemod_embedding::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class Linemod_feature {
public:
    Linemod_feature(){}
    Linemod_feature(cv::Mat& rgb_, cv::Mat depth_, cv::Mat mask_=cv::Mat()):
        rgb(rgb_), depth(depth_), mask(mask_){}
    cv::Mat rgb, depth, mask;
    Info info;
    Linemod_embedding embedding;
    bool constructEmbedding();
    void setEmbedding(Linemod_embedding& embedding_){embedding = std::move(embedding_);}
    float similarity(Linemod_feature& other);

    lchf::Linemod_feature write();
    void read(lchf::Linemod_feature &feature_);
};

template<class matrix_type, class serial_type>
void saveMat(cv::Mat& matrix_i, serial_type* mat_i){
    for(int row=0;row<matrix_i.rows;row++){
        matrix_type* row_p = matrix_i.ptr<matrix_type>(row);
        auto r_p = mat_i->add_row();
        for(int col=0; col<matrix_i.cols; col++){
            r_p->add_value(row_p[col]);
        }
    }
};

template<class mat_type, class serial_type>
void loadMat(cv::Mat& matrix_f, serial_type& mat_f){
    for(int row=0; row<mat_f.row_size();row++){
        float* row_p = matrix_f.ptr<float>(row);
        auto r_p = mat_f.row(row);
        for(int col=0; col<mat_f.row(0).value_size();col++){
            row_p[col] = r_p.value(col);
        }
    }
};

#endif

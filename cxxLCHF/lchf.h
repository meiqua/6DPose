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
    cv::Mat rpy;
    cv::Mat t;
    std::string id;

    void write(lchf::Info* info);
    void read(const lchf::Info &info_);
};

class Linemod_embedding {
public:
    Linemod_embedding():
        weak_threshold(10.0f),
        strong_threshold(55.0f),
        num_features(15),
        distance_threshold(2000),
        difference_threshold(50),
        extract_threshold(2),
        z_check(200){}
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

    std::vector<cv::Mat> rgb_response, dep_response;

    int center_dep;
    int z_check;

    void write(lchf::Linemod_embedding*);
    void read(const lchf::Linemod_embedding& embedding_);
};
inline Linemod_embedding::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class Linemod_feature {
public:
    Linemod_feature(){}
    Linemod_feature(cv::Mat rgb_, cv::Mat depth_):
        rgb(rgb_.clone()), depth(depth_.clone()){}
    Linemod_feature(cv::Mat rgb_, cv::Mat depth_, cv::Mat mask_):
        rgb(rgb_.clone()), depth(depth_.clone()), mask(mask_.clone()){}
    cv::Mat rgb, depth, mask;
    Linemod_embedding embedding;
    std::string name = "linemod";

    bool constructEmbedding();
    bool constructResponse();
    void setEmbedding(Linemod_embedding& embedding_){embedding = std::move(embedding_);}
    float similarity(const Linemod_feature& other) const;

    void write(lchf::Linemod_feature* f);
    void read(const lchf::Linemod_feature &feature_);
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
        mat_type* row_p = matrix_f.ptr<mat_type>(row);
        auto r_p = mat_f.row(row);
        for(int col=0; col<mat_f.row(0).value_size();col++){
            row_p[col] = r_p.value(col);
        }
    }
};

#endif

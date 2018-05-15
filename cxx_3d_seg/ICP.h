///////////////////////////////////////////////////////////////////////////////
///   "Sparse Iterative Closest Point"
///   by Sofien Bouaziz, Andrea Tagliasacchi, Mark Pauly
///   Copyright (C) 2013  LGG, EPFL
///////////////////////////////////////////////////////////////////////////////
///   1) This file contains different implementations of the ICP algorithm.
///   2) This code requires EIGEN and NANOFLANN.
///   3) If OPENMP is activated some part of the code will be parallelized.
///   4) This code is for now designed for 3D registration
///   5) Two main input types are Eigen::Matrix3Xd or Eigen::Map<Eigen::Matrix3Xd>
///////////////////////////////////////////////////////////////////////////////
///   namespace nanoflann: NANOFLANN KD-tree adaptor for EIGEN
///   namespace RigidMotionEstimator: functions to compute the rigid motion
///   namespace SICP: sparse ICP implementation
///   namespace ICP: reweighted ICP implementation
///////////////////////////////////////////////////////////////////////////////
#ifndef ICP_H
#define ICP_H
#include "nanoflann.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "  elasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

///////////////////////////////////////////////////////////////////////////////
/// Compute the rigid motion for point-to-point and point-to-plane distances
namespace RigidMotionEstimator {
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Confidence weights
    Eigen::Affine3f point_to_point(Eigen::Matrix3Xf& X,
                                   Eigen::Matrix3Xf& Y,
                                   Eigen::VectorXf w) {
        /// Normalize weight vector
        Eigen::VectorXf w_normalized = w/w.sum();
        /// De-mean
        Eigen::Vector3f X_mean, Y_mean;

        for(int i=0; i<3; ++i) {
            X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
            Y_mean(i) = (Y.row(i).array()*w_normalized.transpose().array()).sum();
        }
        X.colwise() -= X_mean;
        Y.colwise() -= Y_mean;
        /// Compute transformation
        Eigen::Affine3f transformation;
        Eigen::Matrix3f sigma = X * w_normalized.asDiagonal() * Y.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if(svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            Eigen::Vector3f S = Eigen::Vector3f::Ones();
            S(2) = -1.0;
            transformation.linear().noalias() = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
        } else {
            transformation.linear().noalias() = svd.matrixV()*svd.matrixU().transpose();
        }
        transformation.translation().noalias() = Y_mean - transformation.linear()*X_mean;
        /// Apply transformation
        X = transformation*X;
        /// Re-apply mean
        X.colwise() += X_mean;
        Y.colwise() += Y_mean;
        /// Return transformation
        return transformation;
    }

}
///////////////////////////////////////////////////////////////////////////////
/// ICP implementation using ADMM/ALM/Penalty method
namespace SICP {
    struct Parameters {
        bool use_penalty = false; /// if use_penalty then penalty method else ADMM or ALM (see max_inner)
        float p = 1.0;           /// p norm
        float mu = 10.0;         /// penalty weight
        float alpha = 1.2;       /// penalty increase factor
        float max_mu = 1e5;      /// max penalty
        int max_icp = 100;        /// max ICP iteration
        int max_outer = 20;      /// max outer iteration
        int max_inner = 1;        /// max inner iteration. If max_inner=1 then ADMM else ALM
        float stop = 1e-2;       /// stopping criteria
        bool print_icpn = false;  /// (debug) print ICP iteration 
    };
    /// Shrinkage operator (Automatic loop unrolling using template)
    template<unsigned int I>
    inline float shrinkage(float mu, float n, float p, float s) {
        return shrinkage<I-1>(mu, n, p, 1.0 - (p/mu)*std::pow(n, p-2.0)*std::pow(s, p-1.0));
    }
    template<>
    inline float shrinkage<0>(float, float, float, float s) {return s;}
    /// 3D Shrinkage for point-to-point
    template<unsigned int I>
    inline void shrink(Eigen::Matrix3Xf& Q, float mu, float p) {
        float Ba = std::pow((2.0/mu)*(1.0-p), 1.0/(2.0-p));
        float ha = Ba + (p/mu)*std::pow(Ba, p-1.0);

        #pragma omp parallel for
        for(int i=0; i<Q.cols(); ++i) {
            float n = Q.col(i).norm();
            float w = 0.0;
            if(n > ha) w = shrinkage<I>(mu, n, p, (Ba/n + 1.0)/2.0);
            Q.col(i) *= w;
        }
    }
    /// 1D Shrinkage for point-to-plane
    template<unsigned int I>
    inline void shrink(Eigen::VectorXf& y, float mu, float p) {
        float Ba = std::pow((2.0/mu)*(1.0-p), 1.0/(2.0-p));
        float ha = Ba + (p/mu)*std::pow(Ba, p-1.0);
        #pragma omp parallel for
        for(int i=0; i<y.rows(); ++i) {
            float n = std::abs(y(i));
            float s = 0.0;
            if(n > ha) s = shrinkage<I>(mu, n, p, (Ba/n + 1.0)/2.0);
            y(i) *= s;
        }
    }
    /// Sparse ICP with point to point
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Parameters
    Eigen::Matrix4f point_to_point(Eigen::Matrix3Xf& X,
                        Eigen::Matrix3Xf& Y,
                        Parameters par = Parameters()) {
        Timer timer;

        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
        /// Build kd-tree
        typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix3Xf>  my_kd_tree_t;
        my_kd_tree_t kdtree(Y, 1);
        kdtree.index->buildIndex();

        timer.out("kd-tree");

        /// Buffers
        Eigen::Matrix3Xf Q = Eigen::Matrix3Xf::Zero(3, X.cols());
        Eigen::Matrix3Xf Z = Eigen::Matrix3Xf::Zero(3, X.cols());
        Eigen::Matrix3Xf C = Eigen::Matrix3Xf::Zero(3, X.cols());
        Eigen::Matrix3Xf Xo1 = X;
        Eigen::Matrix3Xf Xo2 = X;
        /// ICP
        for(int icp=0; icp<par.max_icp; ++icp) {
            if(par.print_icpn) std::cout << "Iteration #" << icp << "/" << par.max_icp << std::endl;
            /// Find closest point

            #pragma omp parallel for
            for(int i=0; i<X.cols(); ++i) {
                const size_t num_results = 1;
                std::vector<size_t>   ret_indexes(num_results);
                std::vector<float> out_dists_sqr(num_results);
                nanoflann::KNNResultSet<float> resultSet(num_results);

                resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
                kdtree.index->findNeighbors(resultSet, X.col(i).data(), nanoflann::SearchParams());
                Q.col(i) = Y.col(ret_indexes[0]);
            }

            timer.out("kd search time");

            /// Computer rotation and translation
            float mu = par.mu;
            for(int outer=0; outer<par.max_outer; ++outer) {
                float dual = 0.0;
                for(int inner=0; inner<par.max_inner; ++inner) {
                    /// Z update (shrinkage)
                    Z = X-Q+C/mu;
                    shrink<3>(Z, mu, par.p);
                    /// Rotation and translation update
                    Eigen::Matrix3Xf U = Q+Z-C/mu;
                    Eigen::Affine3f t = RigidMotionEstimator::
                            point_to_point(X, U, Eigen::VectorXf::Ones(X.cols()));
                    Eigen::Matrix4f t2 = t.matrix();
                    transformation = transformation*t2;

                    /// Stopping criteria
                    dual = (X-Xo1).colwise().norm().maxCoeff();
                    Xo1 = X;
                    if(dual < par.stop) break;
                }
                /// C update (lagrange multipliers)
                Eigen::Matrix3Xf P = X-Q-Z;
                if(!par.use_penalty) C.noalias() += mu*P;
                /// mu update (penalty)
                if(mu < par.max_mu) mu *= par.alpha;
                /// Stopping criteria
                float primal = P.colwise().norm().maxCoeff();
                if(primal < par.stop && dual < par.stop) break;
            }
            /// Stopping criteria
            float stop = (X-Xo2).colwise().norm().maxCoeff();
            Xo2 = X;
            if(stop < par.stop) break;
        }
        return transformation;
    }
}
#endif

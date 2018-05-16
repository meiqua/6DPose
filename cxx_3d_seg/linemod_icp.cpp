// copied from ork linemod

#include "linemod_icp.h"
#include "nanoflann.hpp"
#include <limits>

/** A simple vector-of-vectors adaptor for nanoflann, without duplicating the storage.
  *  The i'th vector represents a point in the state space.
  *
  *  \tparam DIM If set to >0, it specifies a compile-time fixed dimensionality for the points in the data set, allowing more compiler optimizations.
  *  \tparam num_t The type of the point coordinates (typically, double or float).
  *  \tparam Distance The distance metric to use: nanoflann::metric_L1, nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc.
  *  \tparam IndexType The type for indices in the KD-tree index (typically, size_t of int)
  */
template <class VectorOfVectorsType, typename num_t = float, int DIM = -1, class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
    typedef KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType,num_t,DIM,Distance> self_t;
    typedef typename Distance::template traits<num_t,self_t>::distance_t metric_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor< metric_t,self_t,DIM,IndexType>  index_t;

    index_t* index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.

    /// Constructor: takes a const ref to the vector of vectors object with the data points
    KDTreeVectorOfVectorsAdaptor(const int /* dimensionality */, const VectorOfVectorsType &mat, const int leaf_max_size = 10) : m_data(mat)
    {
        assert(mat.size() != 0 && 3 != 0);
        const size_t dims = 3;
        if (DIM>0 && static_cast<int>(dims) != DIM)
            throw std::runtime_error("Data set dimensionality does not match the 'DIM' template argument");
        index = new index_t( dims, *this /* adaptor */, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size ) );
        index->buildIndex();
    }

    ~KDTreeVectorOfVectorsAdaptor() {
        delete index;
    }

    const VectorOfVectorsType &m_data;

    /** Query for the \a num_closest closest points to a given point (entered as query_point[0:dim-1]).
      *  Note that this is a short-cut method for index->findNeighbors().
      *  The user can also call index->... methods as desired.
      * \note nChecks_IGNORED is ignored but kept for compatibility with the original FLANN interface.
      */
    inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq, const int nChecks_IGNORED = 10) const
    {
        nanoflann::KNNResultSet<num_t,IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
      * @{ */

    const self_t & derived() const {
        return *this;
    }
    self_t & derived()       {
        return *this;
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return m_data.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, int dim) const {
        return m_data[idx][dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const {
        return false;
    }

    /** @} */

}; // end of KDTreeVectorOfVectorsAdaptor

/** Computes the centroid of 3D points */
void getMean(const std::vector<cv::Vec3f> &pts, cv::Vec3f& centroid)
{
  centroid = cv::Vec3f(0.0f, 0.0f, 0.0f);
  size_t n_points = 0;
  for (std::vector<cv::Vec3f>::const_iterator it = pts.begin(); it != pts.end(); ++it) {
    if (!cv::checkRange(*it))
      continue;
    centroid += (*it);
    ++n_points;
  }

  if (n_points > 0)
  {
    centroid(0) /= float(n_points);
    centroid(1) /= float(n_points);
    centroid(2) /= float(n_points);
  }
}

/** Transforms the point cloud using the rotation and translation */
void transformPoints(const std::vector<cv::Vec3f> &src, std::vector<cv::Vec3f>& dst, const cv::Matx33f &R, const cv::Vec3f &T)
{
  std::vector<cv::Vec3f>::const_iterator it_src = src.begin();
  std::vector<cv::Vec3f>::iterator it_dst = dst.begin();
  for (; it_src != src.end(); ++it_src, ++it_dst) {
    if (!cv::checkRange(*it_src))
      continue;
    (*it_dst) = R * (*it_src) + T;
  }
}

/** Computes the L2 distance between two vectors of 3D points of the same size */
float getL2distClouds(const std::vector<cv::Vec3f> &model, const std::vector<cv::Vec3f> &ref, float &dist_mean, const float mode)
{
  int nbr_inliers = 0;
  int counter = 0;
  float ratio_inliers = 0.0f;

  float dist_expected = dist_mean * 3.0f;
  dist_mean = 0.0f;

  //use the whole region
  std::vector<cv::Vec3f>::const_iterator it_match = model.begin();
  std::vector<cv::Vec3f>::const_iterator it_ref = ref.begin();
  for(; it_match != model.end(); ++it_match, ++it_ref)
  {
    if (!cv::checkRange(*it_ref))
      continue;

    if (cv::checkRange(*it_match))
    {
      float dist = cv::norm(*it_match - *it_ref);
      if ((dist < dist_expected) || (mode == 0))
        dist_mean += dist;
      if (dist < dist_expected)
        ++nbr_inliers;
    }
    ++counter;
  }

  if (counter > 0)
  {
    dist_mean /= float(nbr_inliers);
    ratio_inliers = float(nbr_inliers) / float(counter);
  }
  else
    dist_mean = std::numeric_limits<float>::max();

  return ratio_inliers;
}

/** Refine the object pose by icp (Iterative Closest Point) alignment of two vectors of 3D points.*/
float icpCloudToCloud(const std::vector<cv::Vec3f> &pts_ref_ori, std::vector<cv::Vec3f> &pts_model,
                      cv::Matx33f& R, cv::Vec3f& T, float &px_inliers_ratio, int mode)
{
    typedef KDTreeVectorOfVectorsAdaptor< std::vector<cv::Vec3f>, float >  my_kd_tree_t;
    my_kd_tree_t   mat_index(3 /*dim*/, pts_ref_ori, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // now pts_ref_ori is model points,
    // because linemod_icp is trying to transform model
    std::vector<cv::Vec3f> pts_ref(pts_model.size());
    for(size_t i=0; i<pts_model.size(); i++){
        // do a knn search
        const size_t num_results = 1;
        std::vector<size_t>   ret_indexes(num_results);
        std::vector<float> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<float> resultSet(num_results);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

        float query[3];
        query[0] = pts_model[i](0);
        query[1] = pts_model[i](1);
        query[2] = pts_model[i](2);
        mat_index.index->findNeighbors(resultSet, query, nanoflann::SearchParams(10));
        pts_ref[i] = pts_ref_ori[i];
    }


    //optimal rotation matrix
    cv::Matx33f R_optimal;
    //optimal transformation
    cv::Vec3f T_optimal;

    //the number of desired iterations defined depending on the mode
    int icp_it_th = 35; //maximal number of iterations
    if (mode == 1)
      icp_it_th = 4; //minimal number of iterations
    else if (mode == 2)
      icp_it_th = 4;

    //desired distance between two point clouds
    const float dist_th = 0.012f;
    //The mean distance between the reference and the model point clouds
    float dist_mean = 0.0f;
    px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, mode);
    //The difference between two previously obtained mean distances between the reference and the model point clouds
    float dist_diff = std::numeric_limits<float>::max();

    //the number of performed iterations
    int iter = 0;
    while (( ((dist_mean > dist_th) && (dist_diff > 0.0001f)) || (mode == 1) ) && (iter < icp_it_th))
    {
      ++iter;

      //subsample points from the match and ref clouds
      if (pts_model.empty() || pts_ref.empty())
        continue;

      //compute centroids of each point subset
      cv::Vec3f m_centroid, r_centroid;
      getMean(pts_model, m_centroid);
      getMean(pts_ref, r_centroid);

      //compute the covariance matrix
      cv::Matx33f covariance (0,0,0, 0,0,0, 0,0,0);
      std::vector<cv::Vec3f>::iterator it_s = pts_model.begin();
      std::vector<cv::Vec3f>::const_iterator it_ref = pts_ref.begin();
      for (; it_s < pts_model.end(); ++it_s, ++it_ref)
        covariance += (*it_s) * (*it_ref).t();

      cv::Mat w, u, vt;
      cv::SVD::compute(covariance, w, u, vt);
      //compute the optimal rotation
      R_optimal = cv::Mat(vt.t() * u.t());

      //compute the optimal translation
      T_optimal = r_centroid - R_optimal * m_centroid;
      if (!cv::checkRange(R_optimal) || !cv::checkRange(T_optimal))
        continue;

      //transform the point cloud
      transformPoints(pts_model, pts_model, R_optimal, T_optimal);
      for(size_t i=0; i<pts_model.size(); i++){
          // do a knn search
          const size_t num_results = 1;
          std::vector<size_t>   ret_indexes(num_results);
          std::vector<float> out_dists_sqr(num_results);

          nanoflann::KNNResultSet<float> resultSet(num_results);

          resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

          float query[3];
          query[0] = pts_model[i](0);
          query[1] = pts_model[i](1);
          query[2] = pts_model[i](2);
          mat_index.index->findNeighbors(resultSet, query, nanoflann::SearchParams(10));
          pts_ref[i] = pts_ref_ori[i];
      }

      //compute the distance between the transformed and ref point clouds
      dist_diff = dist_mean;
      px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, mode);
      dist_diff -= dist_mean;

      //update the translation matrix: turn to opposite direction at first and then do translation
      T = R_optimal * T;
      //do translation
      cv::add(T, T_optimal, T);
      //update the rotation matrix
      R = R_optimal * R;
      //std::cout << " it " << iter << "/" << icp_it_th << " : " << std::fixed << dist_mean << " " << d_diff << " " << px_inliers_ratio << " " << pts_model.size() << std::endl;
    }

      //std::cout << " icp " << mode << " " << dist_min << " " << iter << "/" << icp_it_th  << " " << px_inliers_ratio << " " << d_diff << " " << std::endl;
    return dist_mean;
}

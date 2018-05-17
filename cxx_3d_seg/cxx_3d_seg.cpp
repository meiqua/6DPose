#include "cxx_3d_seg.h"


cv::Mat convex_cloud_seg(cv::Mat &rgb, cv::Mat &depth, cv::Mat& sceneK)
{
    auto rgb_slimage = slimage::ConvertToSlimage(rgb);
    auto dep_slimage = slimage::ConvertToSlimage(depth);
    slimage::Image3ub img_color = slimage::anonymous_cast<unsigned char,3>(rgb_slimage);
    slimage::Image1ui16 img_depth = slimage::anonymous_cast<uint16_t,1>(dep_slimage);

    asp::DaspParameters opt_in;
    opt_in.focal_px = sceneK.at<float>(0,0);
    auto test_group = asp::DsapGrouping(img_color, img_depth, opt_in);
    cv::Mat idxs = slimage::ConvertToOpenCv(test_group);
    return idxs;
}

cv::Mat depth2cloud(cv::Mat &depth, cv::Mat &mask, cv::Mat &sceneK)
{
    cv::Mat test_dep;
    depth.copyTo(test_dep, mask);

    cv::Mat sceneCloud;
    cv::rgbd::depthTo3d(test_dep, sceneK, sceneCloud);

    return sceneCloud;
}

cv::Mat pose_estimation(cv::Mat &sceneCloud, std::string ply_model,
                        float LCP_thresh, float ICP_thresh,
                        bool use_pcs, int pcs_seconds, bool use_icp,
                        int cloud_icp_size, int model_icp_size)
{
    cv::Mat result;

    std::vector<GlobalRegistration::Point3D> test_cloud;
    for(auto cloud_iter = sceneCloud.begin<cv::Vec3f>();
        cloud_iter!=sceneCloud.end<cv::Vec3f>(); cloud_iter++){
        if(cv::checkRange(*cloud_iter)){
            GlobalRegistration::Point3D p;
            p.x() = (*cloud_iter)[0]*1000;
            p.y() = (*cloud_iter)[1]*1000;
            p.z() = (*cloud_iter)[2]*1000;
            test_cloud.push_back(p);
        }
    }

    std::vector<GlobalRegistration::Point3D> model_v;
    std::vector<typename GlobalRegistration::Point3D::VectorType> model_n;
    {
        IOManager iom;
        std::vector<Eigen::Matrix2f> tex_coords;
        std::vector<tripple> tris;
        std::vector<std::string> mtls;
        iom.ReadObject(ply_model.c_str(), model_v, tex_coords, model_n, tris, mtls);
    }

    Eigen::Matrix4f	transformation = Eigen::Matrix4f::Identity();
    float score = 0;
    if(use_pcs){
        GlobalRegistration::Match4PCSOptions options;
        options.sample_size = 30;
        options.max_time_seconds = pcs_seconds;
        constexpr GlobalRegistration::Utils::LogLevel loglvl = GlobalRegistration::Utils::Verbose;
        GlobalRegistration::Utils::Logger logger(loglvl);
        GlobalRegistration::MatchSuper4PCS matcher(options, logger);
        score = matcher.ComputeTransformation(model_v, &test_cloud, transformation);
    }

    cv::Mat t1(4,4,CV_32FC1,transformation.data());
    t1 = t1.t();

    cv::Mat t2 = cv::Mat::eye(4,4,CV_32FC1);
    float icp_dist;
    if(use_icp){
        if(model_icp_size > model_v.size()) model_icp_size = model_v.size();
        int model_icp_step = model_v.size()/model_icp_size;
        std::vector<cv::Vec3f> model_v_eigen(model_icp_size);
        for(int i=0; i<model_icp_size; i+=1){
            model_v_eigen[i](0) = model_v[i*model_icp_step].x();
            model_v_eigen[i](1) = model_v[i*model_icp_step].y();
            model_v_eigen[i](2) = model_v[i*model_icp_step].z();
        }

        if(cloud_icp_size > test_cloud.size()) cloud_icp_size = test_cloud.size();
        int cloud_icp_step = test_cloud.size()/cloud_icp_size;
        std::vector<cv::Vec3f> test_cloud_eigen(cloud_icp_size);
        for(int i=0; i<cloud_icp_size; i+=1){
            test_cloud_eigen[i](0) = test_cloud[i*cloud_icp_step].x();
            test_cloud_eigen[i](1) = test_cloud[i*cloud_icp_step].y();
            test_cloud_eigen[i](2) = test_cloud[i*cloud_icp_step].z();
        }

        auto R_real_icp = cv::Matx33f(1,0,0,
                                      0,1,0,
                                      0,0,1);
        auto T_real_icp = cv::Vec3f(0,0,0);
        float px_ratio_match_inliers = 0.0f;
        icp_dist = icpCloudToCloud(model_v_eigen, test_cloud_eigen, R_real_icp,
                                         T_real_icp, px_ratio_match_inliers, 1);

        icp_dist = icpCloudToCloud(model_v_eigen, test_cloud_eigen, R_real_icp,
                                   T_real_icp, px_ratio_match_inliers, 2);

        icp_dist = icpCloudToCloud(model_v_eigen, test_cloud_eigen, R_real_icp,
                                   T_real_icp, px_ratio_match_inliers, 0);

        t2.at<float>(0,3) = T_real_icp[0];
        t2.at<float>(1,3) = T_real_icp[1];
        t2.at<float>(2,3) = T_real_icp[2];
        t2.at<float>(3,3) = 1;

        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                t2.at<float>(i,j) = R_real_icp(i,j);
            }
        }
    }
    if(use_pcs){
        if(score > LCP_thresh){
            if(use_icp){
                if(icp_dist<ICP_thresh){
                    result = (t2*t1).inv();
                    return result;
                }
            }else {
                result = (t2*t1).inv();
                return result;
            }
        }
    }else{
        if(use_icp){
            if(icp_dist<ICP_thresh){
                result = (t2*t1).inv();
                return result;
            }
        }
    }
    return cv::Mat::zeros(4,4,CV_32FC1);
}

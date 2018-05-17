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
                        float LCP_thresh, int pcs_seconds)
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
    {
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

    if(score > LCP_thresh) return t1;

    return cv::Mat::zeros(4,4,CV_32FC1);
}

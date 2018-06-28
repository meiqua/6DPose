#include "cxx_3d_seg.h"
namespace cxx_3d_seg {
convex_result convex_cloud_seg(cv::Mat &rgb, cv::Mat &depth, cv::Mat& sceneK)
{
    convex_result result;

    auto rgb_slimage = slimage::ConvertToSlimage(rgb);
    auto dep_slimage = slimage::ConvertToSlimage(depth);
    slimage::Image3ub img_color = slimage::anonymous_cast<unsigned char,3>(rgb_slimage);
    slimage::Image1ui16 img_depth = slimage::anonymous_cast<uint16_t,1>(dep_slimage);

    asp::DaspParameters opt_in;
    opt_in.focal_px = sceneK.at<float>(0,0);
    opt_in.cx = sceneK.at<float>(0,2);
    opt_in.cy = sceneK.at<float>(1,2);
    slimage::Image3f sli_world;
    slimage::Image3f sli_normal;
    auto test_group = asp::DsapGrouping(img_color, img_depth, opt_in, sli_world, sli_normal);
    cv::Mat idxs = slimage::ConvertToOpenCv(test_group);
    cv::Mat world = slimage::ConvertToOpenCv(sli_world);
    cv::Mat normal = slimage::ConvertToOpenCv(sli_normal);

    // channel order was changed
//    for(int i=0; i<sli_normal.height(); i++){
//        for(int j=0; j<sli_normal.width(); j++){
//            std::cout << sli_normal(j,i)[2] << "\t";
//            std::cout << normal.at<cv::Vec3f>(i,j)[0] << std::endl;
//        }
//    }

    cv::Mat bgr[3];
    std::vector<cv::Mat> bgr_v(3);
    split(world,bgr);
    bgr_v[0] = bgr[2];
    bgr_v[1] = bgr[1];
    bgr_v[2] = bgr[0];
    cv::merge(bgr_v, world);

    split(normal,bgr);
    bgr_v[0] = bgr[2];
    bgr_v[1] = bgr[1];
    bgr_v[2] = bgr[0];
    cv::merge(bgr_v, normal);

    result.normal = normal;
    result.world = world;
    result.indices = idxs;

    return result;
}

cv::Mat pose_estimation(cv::Mat &sceneCloud, std::string ply_model, int pcs_seconds,
                        float LCP_thresh)
{
    cv::Mat result;

    std::vector<GlobalRegistration::Point3D> test_cloud;
    for(auto cloud_iter = sceneCloud.begin<cv::Vec3f>();
        cloud_iter!=sceneCloud.end<cv::Vec3f>(); cloud_iter++){
        if(cv::checkRange(*cloud_iter) &&
                (*cloud_iter)[2] > 0.0001f){
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
        options.sample_size = 200;
        options.max_time_seconds = pcs_seconds;
        constexpr GlobalRegistration::Utils::LogLevel loglvl = GlobalRegistration::Utils::Verbose;
        GlobalRegistration::Utils::Logger logger(loglvl);
        GlobalRegistration::MatchSuper4PCS matcher(options, logger);
        score = matcher.ComputeTransformation(model_v, &test_cloud, transformation);
    }

    Eigen::Matrix4f tran_inv = transformation.inverse();
    cv::Mat t1(4,4,CV_32FC1,tran_inv.data());
    t1 = t1.t();
    std::cout << "final: " << score << std::endl;

    if(score > LCP_thresh) return t1;

    return cv::Mat::zeros(4,4,CV_32FC1);
}

}


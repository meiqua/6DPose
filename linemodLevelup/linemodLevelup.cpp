#include "linemodLevelup.h"
#include <memory>
#include <iostream>
#include "linemod_icp.h"
#include <opencv2/dnn.hpp>
#include <assert.h>
#include <queue>
using namespace std;
using namespace cv;

void poseRefine::process(Mat &sceneDepth, Mat &modelDepth, Mat &sceneK, Mat &modelK,
                         Mat &modelR, Mat &modelT, int detectX, int detectY)
{
    //    sceneDepth.convertTo(sceneDepth, CV_16U);
    //    modelDepth.convertTo(modelDepth, CV_16U);
    assert(sceneDepth.type() == CV_16U);
    assert(sceneK.type() == CV_32F);

    cv::Mat modelMask = modelDepth > 0;
    Mat non0p;
    findNonZero(modelMask, non0p);
    Rect bbox = boundingRect(non0p);

    //crop scene
    int enhancedX = bbox.width / 8 * 0;
    int enhancedY = bbox.height / 8 * 0; // no padding
    //boundary check
    int bboxX1 = detectX - enhancedX;
    if (bboxX1 < 0)
        bboxX1 = 0;
    int bboxX2 = detectX + bbox.width + enhancedX;
    if (bboxX2 > sceneDepth.cols)
        bboxX2 = sceneDepth.cols;
    int bboxY1 = detectY - enhancedY;
    if (bboxY1 < 0)
        bboxY1 = 0;
    int bboxY2 = detectY + bbox.height + enhancedY;
    if (bboxY2 > sceneDepth.rows)
        bboxY2 = sceneDepth.rows;

    int bboxX1_ren = bbox.x - enhancedX;
    if (bboxX1_ren < 0)
        bboxX1_ren = 0;
    int bboxX2_ren = bbox.x + bbox.width + enhancedX;
    if (bboxX2_ren > sceneDepth.cols)
        bboxX2_ren = sceneDepth.cols;
    int bboxY1_ren = bbox.y - enhancedY;
    if (bboxY1_ren < 0)
        bboxY1_ren = 0;
    int bboxY2_ren = bbox.y + bbox.height + enhancedY;
    if (bboxY2_ren > sceneDepth.rows)
        bboxY2_ren = sceneDepth.rows;

    cv::Rect ROI_sceneDepth(bboxX1, bboxY1, bboxX2 - bboxX1, bboxY2 - bboxY1);
    cv::Rect ROI_modelDepth(bboxX1_ren, bboxY1_ren, bboxX2_ren - bboxX1_ren, bboxY2_ren - bboxY1_ren);
    cv::Mat modelCloud_cropped; //rows x cols x 3, cropped
    cv::Mat modelDepth_cropped = modelDepth(ROI_modelDepth);
    cv::rgbd::depthTo3d(modelDepth_cropped, modelK, modelCloud_cropped);

    cv::Mat sceneDepth_cropped = sceneDepth(ROI_sceneDepth);
    cv::Mat sceneCloud_cropped;
    cv::rgbd::depthTo3d(sceneDepth_cropped, sceneK, sceneCloud_cropped);
    //    imshow("rgb_ren_cropped", rgb_ren(ROI_modelDepth));
    //    imshow("rgb_cropped", rgb(ROI_sceneDepth));
    //    waitKey(1000000);

    // get x,y coordinate of obj in scene
    // previous depth-cropped-first version is for icp
    // cropping depth first means we move ROI cloud to center of view
    cv::Mat sceneCloud;
    cv::rgbd::depthTo3d(sceneDepth, sceneK, sceneCloud);

    int smoothSize = 7;
    //boundary check
    int offsetX = ROI_sceneDepth.x + ROI_sceneDepth.width / 2;
    int offsetY = ROI_sceneDepth.y + ROI_sceneDepth.height / 2;
    int startoffsetX1 = offsetX - smoothSize / 2;
    if (startoffsetX1 < 0)
        startoffsetX1 = 0;
    int startoffsetX2 = offsetX + smoothSize / 2;
    if (startoffsetX2 > sceneCloud.cols)
        startoffsetX2 = sceneCloud.cols;
    int startoffsetY1 = offsetY - smoothSize / 2;
    if (startoffsetY1 < 0)
        startoffsetY1 = 0;
    int startoffsetY2 = offsetY + smoothSize / 2;
    if (startoffsetY2 > sceneCloud.rows)
        startoffsetY2 = sceneCloud.rows;

    cv::Vec3f avePoint;
    int count = 0;
    for (auto i = startoffsetX1; i < startoffsetX2; i++)
    {
        for (auto j = startoffsetY1; j < startoffsetY2; j++)
        {
            auto p = sceneCloud.at<cv::Vec3f>(j, i);
            if (checkRange(p))
            {
                avePoint += p;
                count++;
            }
        }
    }
    avePoint /= count;
    modelT.at<float>(0, 0) = avePoint[0] * 1000; // scene cloud unit is meter
    modelT.at<float>(1, 0) = avePoint[1] * 1000;
    // well, it looks stupid
    auto R_real_icp = cv::Matx33f(modelR.at<float>(0, 0), modelR.at<float>(0, 1), modelR.at<float>(0, 2),
                                  modelR.at<float>(1, 0), modelR.at<float>(1, 1), modelR.at<float>(1, 2),
                                  modelR.at<float>(2, 0), modelR.at<float>(2, 1), modelR.at<float>(2, 2));
    auto T_real_icp = cv::Vec3f(modelT.at<float>(0, 0), modelT.at<float>(1, 0), modelT.at<float>(2, 0));

//    std::vector<cv::Vec3f> pts_real_model_temp;
//    std::vector<cv::Vec3f> pts_real_ref_temp;
//    float px_ratio_missing = matToVec(sceneCloud_cropped, modelCloud_cropped, pts_real_ref_temp, pts_real_model_temp);

//    float px_ratio_match_inliers = 0.0f;
//    float icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp,
//                                     T_real_icp, px_ratio_match_inliers, 1);

//    icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp,
//                               T_real_icp, px_ratio_match_inliers, 2);

//    icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp,
//                               T_real_icp, px_ratio_match_inliers, 0);
//    residual = icp_dist;

    R_refined = Mat(R_real_icp);
    t_refiend = Mat(T_real_icp);

}

float poseRefine::getResidual()
{
    return residual;
}

Mat poseRefine::getR()
{
    return R_refined;
}

Mat poseRefine::getT()
{
    return t_refiend;
}

namespace linemodLevelup
{
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label >> cluster;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << cluster << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];
    clusters = fn["clusters"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;
    fs << "clusters" << clusters;
    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

static Rect cropTemplates(std::vector<Template> &templates, int clusters)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            templ.features[j].x -= templ.tl_x;
            templ.features[j].y -= templ.tl_y;
        }
    }

    // third pass: cluster
    bool debug_ = false;

    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        cv::Mat show_templ;
        if(debug_){// show template
            show_templ = cv::Mat(templ.height+1, templ.width+1, CV_8UC3, cv::Scalar(0));
            for(auto f: templ.features){
                cv::circle(show_templ, {f.x, f.y}, 1, {0, 0, 255}, -1);
            }
            cv::imshow("templ", show_templ);
            cv::waitKey(0);
        }

        // make sure one cluster has at least 4 features
        if(templ.features.size() < templ.clusters*4){
            templ.clusters = templ.features.size()/4;
        }
        if(templ.clusters == 0) templ.clusters = 1;

        std::vector<std::vector<int>> k_means_cluster(templ.clusters);
        std::vector<std::vector<int>> k_means_center(templ.clusters);
        std::vector<std::vector<int>> k_means_center_last(templ.clusters);

        // select K init samples
        int steps = int(templ.features.size())/templ.clusters;
        //        if(int(templ.features.size())%templ.clusters>0) steps++;
        for(int j=0; j<templ.clusters; j++){
            std::vector<int> center;
            center.push_back(templ.features[j*steps].x);
            center.push_back(templ.features[j*steps].y);
            k_means_center[j] = center;
        }

        if(debug_){
            for(auto c: k_means_center){
                cv::circle(show_templ, {c[0], c[1]}, 1, {0, 255, 0}, -1);
            }
            cv::imshow("templ", show_templ);
            cv::waitKey(0);
        }

        while(1){
            k_means_center_last = k_means_center;
            k_means_cluster.clear();
            k_means_cluster.resize(templ.clusters);
            // find cloest cluster
            for (int j = 0; j < templ.features.size(); ++j)
            {
                float min_dist = std::numeric_limits<float>::max();
                int closest_k = 0;
                for(int k=0; k<templ.clusters; k++){
                    float dx = (templ.features[j].x - k_means_center[k][0]);
                    float dy = (templ.features[j].y - k_means_center[k][1]);
                    float dist = dx*dx+dy*dy;
                    if(dist<=min_dist){
                        min_dist = dist;
                        closest_k = k;
                    }
                }
                auto& one_means = k_means_cluster[closest_k];
                one_means.push_back(j);
            }
            // recaclulate center
            for(int j=0; j<templ.clusters; j++){
                std::vector<int> center(2, 0);
                auto& one_means = k_means_cluster[j];

                // k-means empty cluster, lucky to find it, an example:
                // http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html
                if(one_means.empty()){
                    one_means.push_back(rand()%templ.features.size());
                }

                for(auto idx: one_means){
                    center[0] += templ.features[idx].x;
                    center[1] += templ.features[idx].y;
                }
                center[0] /= int(one_means.size());
                center[1] /= int(one_means.size());

                k_means_center[j] = center;
            }
            if(k_means_center == k_means_center_last){
                break;
            }
        }
        for(int j=0; j<templ.clusters; j++){
            auto& one_means = k_means_cluster[j];

            assert(one_means.size() < 64 && "too many features for one cluster");

            for(auto idx: one_means){
                templ.features[idx].cluster = j;
            }
        }

        if(debug_){
            show_templ = cv::Mat(templ.height+1, templ.width+1, CV_8UC3, cv::Scalar(0));
            std::vector<cv::Vec3b> color_list(templ.clusters);
            for(auto& color: color_list){
                // color list not too close
                while (1) {
                    bool break_flag = true;
                    color = cv::Vec3b(rand()%255, rand()%255, rand()%255);
                    for(auto& other: color_list){
                        if(color == other) continue;
                        int sum = 0;
                        for(int i=0; i<3; i++){
                            sum += std::abs(color[i] - other[i]);
                        }
                        sum /= 3;
                        if(sum < 60){
                            break_flag = false;
                            break;
                        }
                    }
                    if(break_flag){
                        break;
                    }
                }
            }

            for(auto f: templ.features){
                cv::circle(show_templ, {f.x, f.y}, 1, color_list[f.cluster], -1);
            }
            cv::imshow("templ", show_templ);
            cv::waitKey(0);
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

bool QuantizedPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                               std::vector<Feature> &features,
                                               size_t num_features, float distance)
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;
    //    while (features.size() < num_features)
    while (true) // not limited to num features
    {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int)candidates.size())
        {
            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
            if(distance<5){
                // we don't want two features too close
                break;
            }
        }
    }
    return true;
}

Ptr<Modality> Modality::create(const std::string &modality_type)
{
    if (modality_type == "ColorGradient")
        return makePtr<ColorGradient>();
    else if (modality_type == "DepthNormal")
        return makePtr<DepthNormal>();
    else
        return Ptr<Modality>();
}

/****************************************************************************************\
*                             Color gradient modality                                    *
\****************************************************************************************/

// Forward declaration
void hysteresisGradient(Mat &magnitude, Mat &angle,
                        Mat &ap_tmp, float threshold);

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, float threshold)
{
    magnitude.create(src.size(), CV_32F);

    // Allocate temporary buffers
    Size size = src.size();
    Mat sobel_3dx;              // per-channel horizontal derivative
    Mat sobel_3dy;              // per-channel vertical derivative
    Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
    Mat sobel_dy(size, CV_32F); // maximum vertical derivative
    Mat sobel_ag;               // final gradient orientation (unquantized)
    Mat smoothed;

    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
    Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

    short *ptrx = (short *)sobel_3dx.data;
    short *ptry = (short *)sobel_3dy.data;
    float *ptr0x = (float *)sobel_dx.data;
    float *ptr0y = (float *)sobel_dy.data;
    float *ptrmg = (float *)magnitude.data;

    const int length1 = static_cast<const int>(sobel_3dx.step1());
    const int length2 = static_cast<const int>(sobel_3dy.step1());
    const int length3 = static_cast<const int>(sobel_dx.step1());
    const int length4 = static_cast<const int>(sobel_dy.step1());
    const int length5 = static_cast<const int>(magnitude.step1());
    const int length0 = sobel_3dy.cols * 3;

    for (int r = 0; r < sobel_3dy.rows; ++r)
    {
        int ind = 0;

        for (int i = 0; i < length0; i += 3)
        {
            // Use the gradient orientation of the channel whose magnitude is largest
            int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
            int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
            int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

            if (mag1 >= mag2 && mag1 >= mag3)
            {
                ptr0x[ind] = ptrx[i];
                ptr0y[ind] = ptry[i];
                ptrmg[ind] = (float)mag1;
            }
            else if (mag2 >= mag1 && mag2 >= mag3)
            {
                ptr0x[ind] = ptrx[i + 1];
                ptr0y[ind] = ptry[i + 1];
                ptrmg[ind] = (float)mag2;
            }
            else
            {
                ptr0x[ind] = ptrx[i + 2];
                ptr0y[ind] = ptry[i + 2];
                ptrmg[ind] = (float)mag3;
            }
            ++ind;
        }
        ptrx += length1;
        ptry += length2;
        ptr0x += length3;
        ptr0y += length4;
        ptrmg += length5;
    }

    // Calculate the final gradient orientations
    phase(sobel_dx, sobel_dy, sobel_ag, true);
    hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
}

void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
                        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

class ColorGradientPyramid : public QuantizedPyramid
{
public:
    ColorGradientPyramid(const Mat &src, const Mat &mask,
                         float weak_threshold, size_t num_features,
                         float strong_threshold);

    ColorGradientPyramid(const ColorGradientPyramid& ptr_this, const cv::Mat& mask_crop, const cv::Rect& bbox){
        src = ptr_this.src(bbox).clone();
        pyramid_level = ptr_this.pyramid_level;
        angle = ptr_this.angle(bbox).clone();
        magnitude = ptr_this.magnitude(bbox).clone();
        weak_threshold = ptr_this.weak_threshold;
        num_features = ptr_this.num_features;
        strong_threshold = ptr_this.strong_threshold;

        if(ptr_this.mask.empty()){
            mask = mask_crop(bbox).clone();
        }else{
            cv::bitwise_and(mask_crop(bbox), ptr_this.mask(bbox), mask);
            if(!mask.isContinuous()){
                mask = mask.clone();
            }
        }
    }

    virtual void quantize(Mat &dst) const;

    virtual bool extractTemplate(Template &templ) const;

    virtual void pyrDown();

    virtual void crop_by_mask(const cv::Mat& mask_crop, const cv::Rect& bbox);
    virtual cv::Ptr<QuantizedPyramid> Clone(const cv::Mat& mask_crop, const cv::Rect& bbox){
        // don't know why, pass this pointer directly will erase this's data
        auto qd = makePtr<ColorGradientPyramid>(*this, mask_crop, bbox);
        return qd;}
protected:
    /// Recalculate angle and magnitude images
    void update();

    Mat src;
    Mat mask;

    int pyramid_level;
    Mat angle;
    Mat magnitude;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
};

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, weak_threshold);
}

void ColorGradientPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }

    update();
}

void ColorGradientPyramid::crop_by_mask(const cv::Mat& mask_crop, const cv::Rect& bbox)
{
    src = src(bbox).clone();
    angle = angle(bbox).clone();
    magnitude = magnitude(bbox).clone();
    if(!mask.empty()){
        cv::bitwise_and(mask, mask_crop, mask);
        mask = mask(bbox).clone();
    }else{
        mask = mask_crop(bbox).clone();
    }
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
        subtract(mask, local_mask, local_mask);
    }

    // Create sorted list of all pixels with magnitude greater than a threshold
    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    int nms_kernel_size = 3;
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r)
    {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c)
        {
            if (no_mask || mask_r[c])
            {
                float score = 0;
                if(magnitude_valid.at<uchar>(r, c)>0){
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                            if(r_offset == 0 && c_offset == 0) continue;

                            if(score < magnitude.at<float>(r+r_offset, c+c_offset)){
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                    }

                    if(is_max){
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                                if(r_offset == 0 && c_offset == 0) continue;
                                magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }

                if (score > threshold_sq && angle.at<uchar>(r, c) > 0)
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
                }
            }
        }
    }
    // We require a certain number of features
    if (candidates.size() < num_features){
        std::cout << "ColorGradientPyramid::extractTemplate" << std::endl;
        std::cout << "candidates: " << candidates.size() << std::endl;
        std::cout << "num_features: " << num_features << std::endl;
        return false;
    }

    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient()
    : weak_threshold(10.0f),
      num_features(63),
      strong_threshold(55.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
    return CG_NAME;
}

Ptr<QuantizedPyramid> ColorGradient::processImpl(const std::vector<cv::Mat> &src,
                                                 const Mat &mask) const
{

    auto pd = makePtr<ColorGradientPyramid>(src[0], mask, weak_threshold, num_features, strong_threshold);
    return pd;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}

/****************************************************************************************\
*                               Depth normal modality                                    *
\****************************************************************************************/

// Contains GRANULARITY and NORMAL_LUT
#include "normal_lut.i"

static void accumBilateral(long delta, long i, long j, long *A, long *b, int threshold)
{
    long f = std::abs(delta) < threshold ? 1 : 0;

    const long fi = f * i;
    const long fj = f * j;

    A[0] += fi * i;
    A[1] += fi * j;
    A[3] += fj * j;
    b[0] += fi * delta;
    b[1] += fj * delta;
}

/**
 * \brief Compute quantized normal image from depth image.
 *
 * Implements section 2.6 "Extension to Dense Depth Sensors."
 *
 * \param[in]  src  The source 16-bit depth image (in mm).
 * \param[out] dst  The destination 8-bit image. Each bit represents one bin of
 *                  the view cone.
 * \param distance_threshold   Ignore pixels beyond this distance.
 * \param difference_threshold When computing normals, ignore contributions of pixels whose
 *                             depth difference with the central pixel is above this threshold.
 *
 * \todo Should also need camera model, or at least focal lengths? Replace distance_threshold with mask?
 */
static void quantizedNormals(const Mat &src, Mat &dst, int distance_threshold,
                             int difference_threshold)
{
    dst = Mat::zeros(src.size(), CV_8U);

    const unsigned short *lp_depth = src.ptr<ushort>();
    unsigned char *lp_normals = dst.ptr<uchar>();

    const int l_W = src.cols;
    const int l_H = src.rows;

    const int l_r = 5; // used to be 7
    const int l_offset0 = -l_r - l_r * l_W;
    const int l_offset1 = 0 - l_r * l_W;
    const int l_offset2 = +l_r - l_r * l_W;
    const int l_offset3 = -l_r;
    const int l_offset4 = +l_r;
    const int l_offset5 = -l_r + l_r * l_W;
    const int l_offset6 = 0 + l_r * l_W;
    const int l_offset7 = +l_r + l_r * l_W;

    const int l_offsetx = GRANULARITY / 2;
    const int l_offsety = GRANULARITY / 2;

    for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
    {
        const unsigned short *lp_line = lp_depth + (l_y * l_W + l_r);
        unsigned char *lp_norm = lp_normals + (l_y * l_W + l_r);

        for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
        {
            long l_d = lp_line[0];

            if (l_d < distance_threshold)
            {
                // accum
                long l_A[4];
                l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
                long l_b[2];
                l_b[0] = l_b[1] = 0;
                accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
                accumBilateral(lp_line[l_offset1] - l_d, 0, -l_r, l_A, l_b, difference_threshold);
                accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
                accumBilateral(lp_line[l_offset3] - l_d, -l_r, 0, l_A, l_b, difference_threshold);
                accumBilateral(lp_line[l_offset4] - l_d, +l_r, 0, l_A, l_b, difference_threshold);
                accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
                accumBilateral(lp_line[l_offset6] - l_d, 0, +l_r, l_A, l_b, difference_threshold);
                accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

                // solve
                long l_det = l_A[0] * l_A[3] - l_A[1] * l_A[1];
                long l_ddx = l_A[3] * l_b[0] - l_A[1] * l_b[1];
                long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

                /// @todo Magic number 1150 is focal length? This is something like
                /// f in SXGA mode, but in VGA is more like 530.
                float l_nx = static_cast<float>(1150 * l_ddx);
                float l_ny = static_cast<float>(1150 * l_ddy);
                float l_nz = static_cast<float>(-l_det * l_d);

                float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

                if (l_sqrt > 0)
                {
                    float l_norminv = 1.0f / (l_sqrt);

                    l_nx *= l_norminv;
                    l_ny *= l_norminv;
                    l_nz *= l_norminv;

                    //*lp_norm = fabs(l_nz)*255;

                    int l_val1 = static_cast<int>(l_nx * l_offsetx + l_offsetx);
                    int l_val2 = static_cast<int>(l_ny * l_offsety + l_offsety);
                    int l_val3 = static_cast<int>(l_nz * GRANULARITY + GRANULARITY);

                    *lp_norm = NORMAL_LUT[l_val3][l_val2][l_val1];
                }
                else
                {
                    *lp_norm = 0; // Discard shadows from depth sensor
                }
            }
            else
            {
                *lp_norm = 0; //out of depth
            }
            ++lp_line;
            ++lp_norm;
        }
    }
    medianBlur(dst, dst, 5);
}

class DepthNormalPyramid : public QuantizedPyramid
{
public:
    DepthNormalPyramid(const Mat &src, const Mat &mask,
                       int distance_threshold, int difference_threshold, size_t num_features,
                       int extract_threshold);

    DepthNormalPyramid(const DepthNormalPyramid& ptr_this, const cv::Mat& mask_crop, const cv::Rect& bbox){
        pyramid_level = ptr_this.pyramid_level;
        normal = ptr_this.normal(bbox).clone();
        num_features = ptr_this.num_features;
        extract_threshold = ptr_this.extract_threshold;

        if(ptr_this.mask.empty()){
            mask = mask_crop(bbox).clone();
        }else{
            cv::bitwise_and(mask_crop(bbox), ptr_this.mask(bbox), mask);
            if(!mask.isContinuous()){
                mask = mask.clone();
            }
        }
    }

    virtual void quantize(Mat &dst) const;

    virtual bool extractTemplate(Template &templ) const;

    virtual void pyrDown();

    virtual void crop_by_mask(const cv::Mat& mask_crop, const cv::Rect& bbox);
    virtual cv::Ptr<QuantizedPyramid> Clone(const cv::Mat& mask_crop, const cv::Rect& bbox){
        // don't know why, pass this pointer directly will erase this's data
        auto qd = makePtr<DepthNormalPyramid>(*this, mask_crop, bbox);
        return qd;}
protected:
    Mat mask;

    int pyramid_level;
    Mat normal;

    size_t num_features;
    int extract_threshold;
};

DepthNormalPyramid::DepthNormalPyramid(const Mat &src, const Mat &_mask,
                                       int distance_threshold, int difference_threshold, size_t _num_features,
                                       int _extract_threshold)
    : mask(_mask),
      pyramid_level(0),
      num_features(_num_features),
      extract_threshold(_extract_threshold)
{
    quantizedNormals(src, normal, distance_threshold, difference_threshold);
}

void DepthNormalPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    extract_threshold /= 2;
    ++pyramid_level;

    // In this case, NN-downsample the quantized image
    Mat next_normal;
    Size size(normal.cols / 2, normal.rows / 2);
    resize(normal, next_normal, size, 0.0, 0.0, INTER_NEAREST);
    normal = next_normal;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }
}

void DepthNormalPyramid::crop_by_mask(const Mat &mask_crop, const Rect &bbox)
{
    normal = normal(bbox).clone();
    if(!mask.empty()){
        cv::bitwise_and(mask, mask_crop, mask);
        mask = mask(bbox).clone();
    }else{
        mask = mask_crop(bbox).clone();
    }
}

void DepthNormalPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(normal.size(), CV_8U);
    normal.copyTo(dst, mask);
}

bool DepthNormalPyramid::extractTemplate(Template &templ) const
{
    // Features right on the object border are unreliable
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 2, BORDER_REPLICATE);
    }

    // Compute distance transform for each individual quantized orientation
    Mat temp = Mat::zeros(normal.size(), CV_8U);
    Mat distances[8];
    for (int i = 0; i < 8; ++i)
    {
        temp.setTo(1 << i, local_mask);
        bitwise_and(temp, normal, temp);
        // temp is now non-zero at pixels in the mask with quantized orientation i
        distanceTransform(temp, distances[i], DIST_C, 3);
    }

    // Count how many features taken for each label
    int label_counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Create sorted list of candidate features
    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    for (int r = 0; r < normal.rows; ++r)
    {
        const uchar *normal_r = normal.ptr<uchar>(r);
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0; c < normal.cols; ++c)
        {
            if (no_mask || mask_r[c])
            {
                uchar quantized = normal_r[c];

                if (quantized != 0 && quantized != 255) // background and shadow
                {
                    int label = getLabel(quantized);

                    // Accept if distance to a pixel belonging to a different label is greater than
                    // some threshold. IOW, ideal feature is in the center of a large homogeneous
                    // region.
                    float score = distances[label].at<float>(r, c);
                    if (score >= extract_threshold)
                    {
                        candidates.push_back(Candidate(c, r, label, score));
                        ++label_counts[label];
                    }
                }
            }
        }
    }

    // We require a certain number of features
    if (candidates.size() < num_features){
        std::cout << "DepthNormalPyramid::extractTemplate" << std::endl;
        std::cout << "candidates: " << candidates.size() << std::endl;
        std::cout << "num_features: " << num_features << std::endl;
        return false;
    }

    // Prefer large distances, but also want to collect features over all 8 labels.
    // So penalize labels with lots of candidates.
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        Candidate &c = candidates[i];
        c.score /= (float)label_counts[c.f.label];
    }
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on object cluster for initial distance threshold
    float cluster = no_mask ? (float)normal.total() : (float)countNonZero(local_mask);
    float distance = sqrtf(cluster) / sqrtf((float)num_features) + 1.5f;
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

DepthNormal::DepthNormal()
    : distance_threshold(2000),
      difference_threshold(50),
      num_features(63),
      extract_threshold(2)
{
}

DepthNormal::DepthNormal(int _distance_threshold, int _difference_threshold, size_t _num_features,
                         int _extract_threshold)
    : distance_threshold(_distance_threshold),
      difference_threshold(_difference_threshold),
      num_features(_num_features),
      extract_threshold(_extract_threshold)
{
}

static const char DN_NAME[] = "DepthNormal";

std::string DepthNormal::name() const
{
    return DN_NAME;
}

Ptr<QuantizedPyramid> DepthNormal::processImpl(const std::vector<cv::Mat> &src,
                                               const Mat &mask) const
{
    auto pd = makePtr<DepthNormalPyramid>(src[1], mask, distance_threshold, difference_threshold,
            num_features, extract_threshold);
    return pd;
}

void DepthNormal::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == DN_NAME);

    distance_threshold = fn["distance_threshold"];
    difference_threshold = fn["difference_threshold"];
    num_features = int(fn["num_features"]);
    extract_threshold = fn["extract_threshold"];
}

void DepthNormal::write(FileStorage &fs) const
{
    fs << "type" << DN_NAME;
    fs << "distance_threshold" << distance_threshold;
    fs << "difference_threshold" << difference_threshold;
    fs << "num_features" << int(num_features);
    fs << "extract_threshold" << extract_threshold;
}

/****************************************************************************************\
*                                 Response maps                                          *
\****************************************************************************************/

//static void orUnaligned8u(const uchar *src, const int src_stride,
//                          uchar *dst, const int dst_stride,
//                          const int width, const int height)
//{
//#if CV_SSE2
//    volatile bool haveSSE2 = checkHardwareSupport(CPU_SSE2);
//#if CV_SSE3
//    volatile bool haveSSE3 = checkHardwareSupport(CPU_SSE3);
//#endif
//    bool src_aligned = reinterpret_cast<unsigned long long>(src) % 16 == 0;
//#endif

//    for (int r = 0; r < height; ++r)
//    {
//        int c = 0;

//#if CV_SSE2
//        // Use aligned loads if possible
//        if (haveSSE2 && src_aligned)
//        {
//            for (; c < width - 15; c += 16)
//            {
//                const __m128i *src_ptr = reinterpret_cast<const __m128i *>(src + c);
//                __m128i *dst_ptr = reinterpret_cast<__m128i *>(dst + c);
//                *dst_ptr = _mm_or_si128(*dst_ptr, *src_ptr);
//            }
//        }
//#if CV_SSE3
//        // Use LDDQU for fast unaligned load
//        else if (haveSSE3)
//        {
//            for (; c < width - 15; c += 16)
//            {
//                __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + c));
//                __m128i *dst_ptr = reinterpret_cast<__m128i *>(dst + c);
//                *dst_ptr = _mm_or_si128(*dst_ptr, val);
//            }
//        }
//#endif
//        // Fall back to MOVDQU
//        else if (haveSSE2)
//        {
//            for (; c < width - 15; c += 16)
//            {
//                __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + c));
//                __m128i *dst_ptr = reinterpret_cast<__m128i *>(dst + c);
//                *dst_ptr = _mm_or_si128(*dst_ptr, val);
//            }
//        }
//#endif
//        for (; c < width; ++c)
//            dst[c] |= src[c];

//        // Advance to next row
//        src += src_stride;
//        dst += dst_stride;
//    }
//}


// compiler have done opt for us
static void orUnaligned8u(const uchar *src, const int src_stride,
                          uchar *dst, const int dst_stride,
                          const int width, const int height)
{
    for (int r = 0; r < height; ++r)
    {
        int c = 0;

        for (; c < width; ++c)
            dst[c] |= src[c];

        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }
}

/**
 * \brief Spread binary labels in a quantized image.
 *
 * Implements section 2.3 "Spreading the Orientations."
 *
 * \param[in]  src The source 8-bit quantized image.
 * \param[out] dst Destination 8-bit spread image.
 * \param      T   Sampling step. Spread labels T/2 pixels in each direction.
 */
static void spread(const Mat &src, Mat &dst, int T)
{
    // Allocate and zero-initialize spread (OR'ed) image
    dst = Mat::zeros(src.size(), CV_8U);

    // Fill in spread gradient image (section 2.3)
    for (int r = 0; r < T; ++r)
    {
        int height = src.rows - r;
        for (int c = 0; c < T; ++c)
        {
            orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
                          static_cast<const int>(dst.step1()), src.cols - c, height);
        }
    }
}

// Auto-generated by create_similarity_lut.py
//CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

// 1-->0 2-->1 3-->2
//CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 2, 4, 1, 4, 2, 4, 0, 4, 2, 4, 1, 4, 2, 4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 4, 4, 2, 2, 4, 4, 1, 2, 4, 4, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 4, 2, 4, 1, 4, 2, 4, 0, 4, 2, 4, 1, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 4, 4, 2, 2, 4, 4, 1, 2, 4, 4, 2, 2, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 0, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4};

// 1-->0 2-->1
//CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 3, 4, 1, 4, 3, 4, 0, 4, 3, 4, 1, 4, 3, 4, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 1, 3, 4, 4, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 1, 4, 3, 4, 0, 4, 3, 4, 1, 4, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 4, 4, 3, 3, 4, 4, 1, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 0, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

// 1,2-->0 3-->1
//CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4};

// 1,2-->0 3-->2
//CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 2, 4, 0, 4, 2, 4, 0, 4, 2, 4, 0, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 4, 4, 2, 2, 4, 4, 0, 2, 4, 4, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 4, 2, 4, 0, 4, 2, 4, 0, 4, 2, 4, 0, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 2, 2, 4, 4, 0, 2, 4, 4, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4};

// 1,2,3-->0
//CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4};

/**
 * \brief Precompute response maps for a spread quantized image.
 *
 * Implements section 2.4 "Precomputing Response Maps."
 *
 * \param[in]  src           The source 8-bit spread quantized image.
 * \param[out] response_maps Vector of 8 response maps, one for each bit label.
 */
static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);

        for (int c = 0; c < src.cols; ++c)
        {
            // Least significant 4 bits of spread image pixel
            lsb4_r[c] = src_r[c] & 15;
            // Most significant 4 bits, right-shifted to be in [0, 16)
            msb4_r[c] = (src_r[c] & 240) >> 4;
        }
    }

#if CV_SSSE3
    volatile bool haveSSSE3 = checkHardwareSupport(CV_CPU_SSSE3);
    if (haveSSSE3)
    {
        const __m128i *lut = reinterpret_cast<const __m128i *>(SIMILARITY_LUT);
        for (int ori = 0; ori < 8; ++ori)
        {
            __m128i *map_data = response_maps[ori].ptr<__m128i>();
            __m128i *lsb4_data = lsb4.ptr<__m128i>();
            __m128i *msb4_data = msb4.ptr<__m128i>();

            // Precompute the 2D response map S_i (section 2.4)
            for (int i = 0; i < (src.rows * src.cols) / 16; ++i)
            {
                // Using SSE shuffle for table lookup on 4 orientations at a time
                // The most/least significant 4 bits are used as the LUT index
                __m128i res1 = _mm_shuffle_epi8(lut[2 * ori + 0], lsb4_data[i]);
                __m128i res2 = _mm_shuffle_epi8(lut[2 * ori + 1], msb4_data[i]);

                // Combine the results into a single similarity score
                map_data[i] = _mm_max_epu8(res1, res2);
            }
        }
    }
    else
#endif
    {
        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 8; ++ori)
        {
            uchar *map_data = response_maps[ori].ptr<uchar>();
            uchar *lsb4_data = lsb4.ptr<uchar>();
            uchar *msb4_data = msb4.ptr<uchar>();
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
            const uchar *lut_hi = lut_low + 16;

            for (int i = 0; i < src.rows * src.cols; ++i)
            {
                map_data[i] = std::max(lut_low[lsb4_data[i]], lut_hi[msb4_data[i]]);
            }
        }
    }
}

/**
 * \brief Convert a response map to fast linearized ordering.
 *
 * Implements section 2.5 "Linearizing the Memory for Parallelization."
 *
 * \param[in]  response_map The 2D response map, an 8-bit image.
 * \param[out] linearized   The response map in linearized order. It has T*T rows,
 *                          each of which is a linear memory of length (W/T)*(H/T).
 * \param      T            Sampling step.
 */
static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];
            }
        }
    }
}
/****************************************************************************************\
*                               Linearized similarities                                  *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
                                               const Feature &f, int T, int W)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

static void similarity(std::vector<uint16_t> &cluster_counts, const std::vector<Mat> &linear_memories, const Template &templ,
                       std::vector<Mat> &dst_vec, Size size, int T)
{
    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    dst_vec.resize(templ.clusters);
    for(int i=0; i<templ.clusters; i++){
        dst_vec[i] = Mat::zeros(H, W, CV_8U);
    }
    cluster_counts.resize(templ.clusters);
    std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];

        cluster_counts[f.cluster] += 1;
        uchar *dst_ptr = dst_vec[f.cluster].ptr<uchar>();

        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;
        // Process responses 16 at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
            for ( ; j < template_positions - 15; j += 16)
            {
                __m128i responses = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
                __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
                *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
            }
        }
        else
#endif
            if (haveSSE2)
            {
                // Fall back to MOVDQU
                for ( ; j < template_positions - 15; j += 16)
                {
                    __m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
                    __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
                    *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
                }
            }
#endif
        for ( ; j < template_positions; ++j)
            dst_ptr[j] = uchar(dst_ptr[j] + lm_ptr[j]);
    }
}

static void similarityLocal(std::vector<uint16_t> &cluster_counts, const std::vector<Mat> &linear_memories, const Template &templ,
                            std::vector<Mat> &dst_vec, Size size, int T, Point center)
{
    int W = size.width / T;

    dst_vec.resize(templ.clusters);
    for(int i=0; i<templ.clusters; i++){
        dst_vec[i] = Mat::zeros(16, 16, CV_8U);
    }
    cluster_counts.resize(templ.clusters);
    std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);

#endif

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        cluster_counts[f.cluster] += 1;
        __m128i *dst_ptr_sse = dst_vec[f.cluster].ptr<__m128i>();

        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Process whole row at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
                dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
            if (haveSSE2)
            {
                // Fall back to MOVDQU
                for (int row = 0; row < 16; ++row)
                {
                    __m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
                    dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                    lm_ptr += W; // Step to next row
                }
            }
            else
#endif
            {
                uchar* dst_ptr = dst_vec[f.cluster].ptr<uchar>();
                for (int row = 0; row < 16; ++row)
                {
                    for (int col = 0; col < 16; ++col)
                        dst_ptr[col] = uchar(dst_ptr[col] + lm_ptr[col]);
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
    }

}

/****************************************************************************************\
*                               High-level Detector API                                  *
\****************************************************************************************/

Detector::Detector()
{
    std::vector<Ptr<Modality>> modalities;
    modalities.push_back(makePtr<ColorGradient>());
    modalities.push_back(makePtr<DepthNormal>());
    this->modalities = modalities;
    pyramid_levels = 2;
    T_at_level.push_back(5);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T, int clusters_)
{
    std::vector<Ptr<Modality>> modalities;
    modalities.push_back(makePtr<ColorGradient>());
    modalities.push_back(makePtr<DepthNormal>());
    this->modalities = modalities;
    pyramid_levels = T.size();
    T_at_level = T;
    clusters = clusters_;
}

Detector::Detector(int num_features, std::vector<int> T, int clusters_)
{
    std::vector<Ptr<Modality>> modalities;
    modalities.push_back(makePtr<ColorGradient>(10.0f, num_features, 55.0f));
    modalities.push_back(makePtr<DepthNormal>(2000, 50, num_features, 2));
    this->modalities = modalities;
    pyramid_levels = T.size();
    T_at_level = T;
    clusters = clusters_;
}

Detector::Detector(const std::vector<Ptr<Modality>> &_modalities,
                   const std::vector<int> &T_pyramid)
    : modalities(_modalities),
      pyramid_levels(static_cast<int>(T_pyramid.size())),
      T_at_level(T_pyramid)
{
}

static std::vector<cv::Mat> crop_to_same_depth_parts(const cv::Mat &depth, const std::vector<int>& dep_anchors,
                                                     const int dep_range, std::vector<int>& dep_templs, int T=8){
    dep_templs.clear();

    const int cell_size = T;
    const int seed_stride = 3;
    assert(depth.rows%cell_size==0 && depth.cols%cell_size==0 && "depth rows&cols should be 8n");

    cv::Mat cell_depth = cv::Mat(depth.rows/cell_size, depth.cols/cell_size, depth.type(), cv::Scalar(0));
    for(int r=0; r<cell_depth.rows; r++){
        for(int c=0; c<cell_depth.cols; c++){
            cv::Rect roi = cv::Rect(c*cell_size, r*cell_size, cell_size, cell_size);
            cell_depth.at<uint16_t>(r, c) = uint16_t(cv::sum(depth(roi))[0]/cell_size/cell_size);
        }
    }

    const int cols = cell_depth.cols;
    const int rows = cell_depth.rows;
    auto rc2id = [cols](int r, int c){return (r*cols+c);};
    auto id2rc = [cols](int id, int& r, int& c){r = id/cols; c=id%cols;};
    auto valid_check = [rows, cols](int r, int c, const cv::Mat& mask){
        if(r>=0 && r<rows && c>=0 && c<cols){
            if(mask.at<uchar>(r,c)==0){
                return true;
            }
        }
        return false;
    };

    std::vector<cv::Mat> masks_seeds;
    std::vector<int> seeds_avg_dep;
    for(int r=1; r<cell_depth.rows-1; r+=seed_stride){
        for(int c=1; c<cell_depth.cols-1; c+=seed_stride){
            cv::Mat masks_seed = cv::Mat(cell_depth.size(), CV_8UC1, cv::Scalar(0));
            masks_seed.at<uchar>(r, c) = 255;

            {  // bread first extend
                std::queue<int> bfs_queue;
                bfs_queue.push(rc2id(r,c));
                int current_min = cell_depth.at<uint16_t>(r,c);
                int current_max = current_min;
                int total_dep = current_max;
                int total_dep_count = 1;
                while (!bfs_queue.empty()) {
                    int front_r, front_c;
                    id2rc(bfs_queue.front(), front_r, front_c);

                    for(int offset_r=-1; offset_r<=1; offset_r++){
                        for(int offset_c=-1; offset_c<=1; offset_c++){
                            if(offset_r==0 && offset_c==0) continue;
                            int next_r = front_r + offset_r;
                            int next_c = front_c + offset_c;

                            if(!valid_check(next_r, next_c, masks_seed)) continue;

                            int depth_to_add = cell_depth.at<uint16_t>(next_r, next_c);
                            int local_min = current_min;
                            int local_max = current_max;
                            if(depth_to_add>local_max) local_max = depth_to_add;
                            if(depth_to_add<local_min) local_min = depth_to_add;

                            if(local_max-local_min<dep_range){
                                current_max = local_max;
                                current_min = local_min;
                                masks_seed.at<uchar>(next_r, next_c) = 255;
                                total_dep += depth_to_add;
                                total_dep_count++;
                                bfs_queue.push(rc2id(next_r, next_c));
                            }
                        }
                    }
                    bfs_queue.pop();
                }
                seeds_avg_dep.push_back(total_dep/total_dep_count);
            }
            cv::dilate(masks_seed, masks_seed, cv::Mat());
            masks_seeds.push_back(masks_seed);
        }
    }

    int min_anchor=*std::min_element(dep_anchors.begin(), dep_anchors.end());
    int max_anchor=*std::max_element(dep_anchors.begin(), dep_anchors.end());
    auto find_closest_dep_idx = [&dep_anchors, min_anchor, max_anchor](int ave_dep){
        if(ave_dep<min_anchor || ave_dep>max_anchor){
            return -1;
        }

        int cloest_dep_idx = 0;
        int min_dist = std::numeric_limits<int>::max();
        for(int i=0; i<dep_anchors.size(); i++){
            auto& dep = dep_anchors[i];
            int dist = std::abs(dep-ave_dep);
            if(dist<min_dist){
                min_dist = dist;
                cloest_dep_idx = i;
            }
        }
        return cloest_dep_idx;
    };

    std::vector<cv::Mat> masks(dep_anchors.size());
    for(size_t i=0; i<seeds_avg_dep.size(); i++){
        int ave_dep = seeds_avg_dep[i];
        int closest_anchor_idx = find_closest_dep_idx(ave_dep);
        if(closest_anchor_idx<0) continue;

        auto& mask = masks[closest_anchor_idx];
        if(mask.empty()) mask = cv::Mat(masks_seeds[i].size(), CV_8UC1, cv::Scalar(0));

        cv::bitwise_or(mask, masks_seeds[i], mask);
    }

    std::vector<cv::Mat> masks_final;
    for(int mask_iter=0; mask_iter<masks.size(); mask_iter++){
        cv::Mat cell_mask = masks[mask_iter];
        if(cell_mask.empty()) continue;
        int dep_anchor = dep_anchors[mask_iter];

        cv::Mat labels;
        cv::connectedComponents(cell_mask, labels, 8, CV_16UC1);

        int max_label = 0;
        for(int r=0; r<labels.rows; r++){
            uint16_t* label_row = labels.ptr<uint16_t>(r);
            for(int c=0; c<labels.cols; c++){
                int label = label_row[c];
                if(label>max_label){
                    max_label = label;
                }
            }
        }
        assert(max_label>0 && "What? it's impossible.");

        for(int i=1; i<=max_label; i++){
            cv::Mat connected_mask = (labels == i);
            cv::resize(connected_mask, connected_mask,
            {connected_mask.cols*cell_size, connected_mask.rows*cell_size}, 0, 0, cv::INTER_NEAREST);

            masks_final.push_back(connected_mask);
            dep_templs.push_back(dep_anchor);
        }
    }

    return masks_final;
}

std::vector<Match> Detector::match(const std::vector<Mat> &sources, float threshold, float active_ratio,
                                   const std::vector<std::string> &class_ids,
                                   const std::vector<int>& dep_anchors, const int dep_range,
                                   const std::vector<Mat> &masks_ori) const
{
    std::vector<Match> matches_final;
    CV_Assert(sources.size() == modalities.size());

    std::vector<Ptr<QuantizedPyramid>> quantizers;

    std::vector<Mat> masks = masks_ori;
    if(masks.size() == 1 && modalities.size() > 1){
        for (int i = 1; i < (int)modalities.size(); ++i){
            masks.push_back(masks[0]);
        }
    }
    for (int i = 0; i < (int)modalities.size(); ++i)
    {
        Mat mask, source;
        source = sources[i];
        if (!masks.empty())
        {
            CV_Assert(masks.size() == modalities.size());
            mask = masks[i];
        }
        CV_Assert(mask.empty() || mask.size() == source.size());
        quantizers.push_back(modalities[i]->process(sources, mask));
    }
    // whole image quantizers OK

    auto find_matches = [&](std::vector<Ptr<QuantizedPyramid>>& quantizers, const std::vector<std::string> &class_ids){
        std::vector<Match> matches;
        LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                       std::vector<LinearMemories>(modalities.size(), LinearMemories(8)));

        // For each pyramid level, precompute linear memories for each modality
        std::vector<Size> sizes;
        for (int l = 0; l < pyramid_levels; ++l)
        {
            int T = T_at_level[l];
            std::vector<LinearMemories> &lm_level = lm_pyramid[l];

            if (l > 0)
            {
                for (int i = 0; i < (int)quantizers.size(); ++i)
                    quantizers[i]->pyrDown();
            }

            Mat quantized, spread_quantized;
            std::vector<Mat> response_maps;
            for (int i = 0; i < (int)quantizers.size(); ++i)
            {
                quantizers[i]->quantize(quantized);
                spread(quantized, spread_quantized, T);
                computeResponseMaps(spread_quantized, response_maps);

                LinearMemories &memories = lm_level[i];
                for (int j = 0; j < 8; ++j)
                    linearize(response_maps[j], memories[j], T);
            }
            sizes.push_back(quantized.size());
        }

        if (class_ids.empty())
        {
            // Match all templates
            TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
            for (; it != itend; ++it)
                matchClass(lm_pyramid, sizes, threshold, active_ratio, matches, it->first, it->second);
        }
        else
        {
            // Match only templates for the requested class IDs
            for (int i = 0; i < (int)class_ids.size(); ++i)
            {
                TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
                if (it != class_templates.end())
                    matchClass(lm_pyramid, sizes, threshold, active_ratio, matches, it->first, it->second);
            }
        }

        // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
        std::sort(matches.begin(), matches.end());
        std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
        matches.erase(new_end, matches.end());
        return matches;
    };

    if(dep_anchors.empty()){
        // Initialize each modality with our sources
        matches_final = find_matches(quantizers, class_ids);
    }else{
        std::vector<int> dep_templs;
        auto mask_vec = crop_to_same_depth_parts(sources[1], dep_anchors,
                dep_range, dep_templs, T_at_level.back());
        for(size_t i=0; i<mask_vec.size(); i++){
            cv::Mat mask = mask_vec[i];
            int dep_templ = dep_templs[i];

            cv::Rect bbox;
            {
                cv::Mat Points;
                cv::findNonZero(mask,Points);
                bbox=cv::boundingRect(Points);

                // discard small area
                if(bbox.width<=32 || bbox.height<=32) continue;

                if(bbox.width%16!=0){
                    bbox.width += (16 - bbox.width%16);
                    if(bbox.width + bbox.x > mask.cols){
                        bbox.x -= (16 - bbox.width%16);
                        if(bbox.x < 0) bbox.x = 0;
                    }
                }
                if(bbox.height%16!=0){
                    bbox.height += (16 - bbox.height%16);
                    if(bbox.height + bbox.y > mask.rows){
                        bbox.y -= (16 - bbox.height%16);
                        if(bbox.y < 0) bbox.y = 0;
                    }
                }
//                assert(bbox.height%16==0 && bbox.width%16==0);
            }

            // parts quant, clone to avoid writing whole quant
            std::vector<Ptr<QuantizedPyramid>> quantizers_part;
            for(auto& quant: quantizers){
                auto pd = quant->Clone(mask, bbox);
                quantizers_part.push_back(pd);
            }

            std::vector<std::string> ids_with_dep;
            for(const auto& class_id: class_ids){
                auto idx = class_id.find_last_of('_');
                ids_with_dep.push_back(class_id.substr(0, idx) +
                                             "_" + std::to_string(dep_templ));
            }
            ids_with_dep.erase(unique(ids_with_dep.begin(),
                                              ids_with_dep.end()), ids_with_dep.end() );

            auto matches = find_matches(quantizers_part, ids_with_dep);
            for(auto& match: matches){
                match.x += bbox.x;
                match.y += bbox.y;
            }
            matches_final.insert(matches_final.end(), matches.begin(), matches.end());

            bool debug_ = false;
            if(debug_){
                auto view_dep = [](cv::Mat dep){
                    cv::Mat map = dep;
                    double min;
                    double max;
                    cv::minMaxIdx(map, &min, &max);
                    cv::Mat adjMap;
                    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
                    cv::Mat falseColorsMap;
                    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
                    return falseColorsMap;
                };

                std::cout << "dep_templ: " << dep_templ << std::endl;

                cv::Mat draw = sources[0].clone();
                cv::Mat depth = view_dep(sources[1]);

                cv::rectangle(draw, bbox.tl(), bbox.br(), {0, 255, 0}, 2);
                cv::rectangle(depth, bbox.tl(), bbox.br(), {0, 255, 0}, 2);

                cv::Mat mask_rgb;
                sources[0].copyTo(mask_rgb, mask);

                for(auto& match: matches){
                    int r = 40;
                    cout << "x: " << match.x << "\ty: " << match.y
                         << "\tsimilarity: "<< match.similarity <<endl;
                    cv::circle(draw, cv::Point(match.x,match.y), 2, cv::Scalar(255, 0 ,255), -1);
                    cv::putText(draw, to_string(int(round(match.similarity))),
                                cv::Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 1.4, cv::Scalar(0,255,255));
                }

                cv::imshow("depth",depth);
                cv::imshow("mask_rgb", mask_rgb);
                cv::imshow("rgb", draw);

                cv::waitKey(0);
            }
        }
    }
    return matches_final;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

static cv::Mat add_16u_8u(cv::Mat& simi_sum, const cv::Mat& simi, const cv::Mat& mask){
    cv::Mat active_score_local;
    simi.copyTo(active_score_local, mask);
    active_score_local.convertTo(active_score_local, CV_16UC1);
    simi_sum += active_score_local;
    return simi_sum;
};

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
                          const std::vector<Size> &sizes,
                          float threshold, float active_ratio, std::vector<Match> &matches,
                          const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids) const
{
    #pragma omp parallel for
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        const TemplatePyramid &tp = template_pyramids[template_id];
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();

        std::vector<Match> candidates;
        { // Compute similarity maps for each modality at lowest pyramid level
            std::vector<std::vector<Mat>> similarities(modalities.size());
            std::vector<std::vector<uint16_t>> cluster_counts(modalities.size());

            int lowest_start = static_cast<int>(tp.size() - modalities.size());
            int lowest_T = T_at_level.back();

            std::vector<int> total_counts(modalities.size(), 0);
            for (int i = 0; i < (int)modalities.size(); ++i)
            {
                const Template &templ = tp[lowest_start + i];
                total_counts[i] = templ.clusters;
                similarity(cluster_counts[i], lowest_lm[i], templ, similarities[i], sizes.back(), lowest_T);
            }
            cv::Mat nms_candidates = cv::Mat(similarities[0][0].rows, similarities[0][0].cols, CV_32FC1,
                    cv::Scalar(std::numeric_limits<float>::max()));

            for (int i = 0; i < (int)modalities.size(); ++i)
            { // min score of two modalities
                Mat active_count_1mod = Mat::zeros(similarities[0][0].rows, similarities[0][0].cols, CV_8UC1);
                Mat active_score_1mod = Mat::zeros(similarities[0][0].rows, similarities[0][0].cols, CV_16UC1);
                Mat active_feat_num_1mod = Mat::zeros(similarities[0][0].rows, similarities[0][0].cols, CV_16UC1);

                for(int j=0; j<similarities[i].size(); j++){
                    auto& simi = similarities[i][j];
                    int feat_count = cluster_counts[i][j];

                    int raw_thresh = int((threshold)*(4 * feat_count)/100.0f);

                    cv::Mat active_mask = simi > raw_thresh;

                    active_count_1mod += active_mask/255;

                    add_16u_8u(active_score_1mod, simi, active_mask);

                    cv::Mat active_unit;
                    active_mask.convertTo(active_unit, CV_16UC1);
                    active_feat_num_1mod += active_unit/255*feat_count;
                }

                for (int r = 0; r < active_count_1mod.rows; ++r)
                {
                    ushort *raw_score = active_score_1mod.ptr<ushort>(r);
                    ushort *active_feats = active_feat_num_1mod.ptr<ushort>(r);
                    uchar* active_parts = active_count_1mod.ptr<uchar>(r);

                    float* nms_row = nms_candidates.ptr<float>(r);
                    for (int c = 0; c < active_count_1mod.cols; ++c)
                    {
                        float score = 100.0f/4*raw_score[c]/active_feats[c];
                        if (active_parts[c] > int(total_counts[i]*active_ratio) && score>threshold)
                        {
                            if(score < nms_row[c])
                                nms_row[c] = score;
                        }else{
                            nms_row[c] = 0;
                        }
                    }
                }
            }

            // Find initial matches, nms
            int nms_kernel_size = 3;
            for (int r = nms_kernel_size/2; r < similarities[0][0].rows-nms_kernel_size/2; ++r)
            {
                float* nms_row = nms_candidates.ptr<float>(r);
                for (int c = nms_kernel_size/2; c < similarities[0][0].cols-nms_kernel_size/2; ++c)
                {
                    float score = nms_row[c];
                    if(score<=0) continue;

                    bool is_max = true;
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                            if(r_offset == 0 && c_offset == 0) continue;

                            if(score < nms_candidates.at<float>(r+r_offset, c+c_offset)){
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                    }

                    if(is_max){
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                                if(r_offset == 0 && c_offset == 0) continue;
                                nms_candidates.at<float>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }
            }

            for (int r = 0; r < similarities[0][0].rows; ++r)
            {
                float* nms_row = nms_candidates.ptr<float>(r);
                for (int c = 0; c < similarities[0][0].cols; ++c){
                    if(nms_row[c]>0){
                        int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                        int x = c * lowest_T + offset;
                        int y = r * lowest_T + offset;
                        candidates.push_back(Match(x, y, nms_row[c], class_id, static_cast<int>(template_id)));
                    }
                }
            }
        }

        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l)
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];
            int start = static_cast<int>(l * modalities.size());
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            const int local_size = 16;
            std::vector<std::vector<Mat>> similarities2(modalities.size());
            std::vector<std::vector<uint16_t>> cluster_counts2(modalities.size());

            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                std::vector<int> total_counts2(modalities.size());

                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                for (int i = 0; i < (int)modalities.size(); ++i)
                {
                    const Template &templ = tp[start + i];
                    total_counts2[i] = templ.clusters;
                    similarityLocal(cluster_counts2[i], lms[i], templ, similarities2[i], size, T, Point(x, y));
                }

                float best_score = 0;
                int best_r = -1, best_c = -1;

                std::vector<cv::Mat> score_2mod(modalities.size());
                for (int i = 0; i < (int)modalities.size(); ++i){
                    Mat active_count2 = Mat::zeros(local_size, local_size, CV_8UC1);
                    Mat active_feat_num2 = Mat::zeros(local_size, local_size, CV_16UC1);
                    Mat active_score2 = Mat::zeros(local_size, local_size, CV_16UC1);

                    for(int j=0; j<total_counts2[i]; j++){
                        auto& simi = similarities2[i][j];
                        uint16_t feat_count = cluster_counts2[i][j];

                        uint16_t raw_thresh = uint16_t((threshold)*(4 * feat_count)/100.0f);
                        cv::Mat active_mask = simi > raw_thresh;

                        active_count2 += active_mask/255;

                        add_16u_8u(active_score2, simi, active_mask);

                        cv::Mat active_unit;
                        active_mask.convertTo(active_unit, CV_16UC1);
                        active_feat_num2 += active_unit/255*feat_count;
                    }

                    score_2mod[i] = cv::Mat(local_size, local_size, CV_32FC1, cv::Scalar(0));
                    for (int r = 0; r < local_size; ++r)
                    {
                        ushort* active_score2_row = active_score2.ptr<ushort>(r);
                        ushort* active_feat_num2_row = active_feat_num2.ptr<ushort>(r);
                        uchar* active_count2_row = active_count2.ptr<uchar>(r);
                        float* score_2mod_row = score_2mod[i].ptr<float>(r);
                        for (int c = 0; c < local_size; ++c)
                        {
                            if (active_count2_row[c]>(total_counts2[i]*active_ratio))
                            {
                                score_2mod_row[c] = 100.0f/4*
                                        active_score2_row[c]
                                        /active_feat_num2_row[c];
                            }
                        }
                    }
                }

                cv::Mat min_score(score_2mod[0].size(), CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));
                for(auto& score: score_2mod){
                    min_score = cv::min(min_score, score);
                }

                for (int r = 0; r < local_size; ++r)
                {
                    float* score_2mod_row = min_score.ptr<float>(r);
                    for (int c = 0; c < local_size; ++c)
                    {
                        if(score_2mod_row[c] > best_score){
                            best_score = score_2mod_row[c];
                            best_c = c;
                            best_r = r;
                        }
                    }
                }

                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                                  MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());
        }

        #pragma omp critical
        {
            matches.insert(matches.end(), candidates.begin(), candidates.end());
        }
    }
}

int Detector::addTemplate(const std::vector<Mat> &sources, const std::string &class_id,
                          const Mat &object_mask)
{
    int num_modalities = static_cast<int>(modalities.size());
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(num_modalities * pyramid_levels);

    // For each modality...
    for (int i = 0; i < num_modalities; ++i)
    {
        // Extract a template at each pyramid level
        Ptr<QuantizedPyramid> qp = modalities[i]->process(sources, object_mask);
        for (int l = 0; l < pyramid_levels; ++l)
        {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0)
                qp->pyrDown();

            bool success = qp->extractTemplate(tp[l * num_modalities + i]);
            if (!success)
                return -1;
        }
    }

    Rect bb = cropTemplates(tp, clusters);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}
const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    clusters = fn["clusters"];
    fn["T"] >> T_at_level;

    modalities.clear();
    FileNode modalities_fn = fn["modalities"];
    FileNodeIterator it = modalities_fn.begin(), it_end = modalities_fn.end();
    for (; it != it_end; ++it)
    {
        modalities.push_back(Modality::create(*it));
    }
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;
    fs << "clusters" << clusters;

    fs << "modalities"
       << "[";
    for (int i = 0; i < (int)modalities.size(); ++i)
    {
        fs << "{";
        modalities[i]->write(fs);
        fs << "}";
    }
    fs << "]"; // modalities
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Verify compatible with Detector settings
    FileNode mod_fn = fn["modalities"];
    CV_Assert(mod_fn.size() == modalities.size());
    FileNodeIterator mod_it = mod_fn.begin(), mod_it_end = mod_fn.end();
    int i = 0;
    for (; mod_it != mod_it_end; ++mod_it, ++i)
        CV_Assert(modalities[i]->name() == (String)(*mod_it));
    CV_Assert((int)fn["pyramid_levels"] == pyramid_levels);

    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id][idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    fs << "modalities"
       << "[:";
    for (size_t i = 0; i < modalities.size(); ++i)
        fs << modalities[i]->name();
    fs << "]"; // modalities
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.size(); ++j)
        {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,
                           const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

} // namespace linemodLevelup

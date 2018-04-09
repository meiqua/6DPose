#include "lchf.h"
#include <memory>
#include <chrono>
#include <assert.h>
using namespace std;
using namespace cv;

static inline int getLabel(int quantized)
{
  switch (quantized)
  {
    case 1:   return 0;
    case 2:   return 1;
    case 4:   return 2;
    case 8:   return 3;
    case 16:  return 4;
    case 32:  return 5;
    case 64:  return 6;
    case 128: return 7;
    default:
      CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
      return -1; //avoid warning
  }
}

static Rect cropTemplates(std::vector<Linemod_embedding::element>& elements)
{
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();

  for (auto& element: elements){
    int x = element.x;
    int y = element.y;
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
  }

  /// @todo Why require even min_x, min_y?
  if (min_x % 2 == 1) --min_x;
  if (min_y % 2 == 1) --min_y;

  for (auto& element: elements){
      element.x -= min_x;
      element.y -= min_y;
  }

  return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

static void hysteresisGradient(Mat &magnitude, Mat &quantized_angle, Mat &angle, float threshold)
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
    uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
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
    float* mag_r = magnitude.ptr<float>(r);

    for (int c = 1; c < angle.cols - 1; ++c)
    {
      if (mag_r[c] > threshold)
      {
  // Compute histogram of quantized bins in 3x3 patch around pixel
        int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        uchar* patch3x3_row = &quantized_unfiltered(r-1, c-1);
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

static void quantizedOrientations(const Mat &src, Mat &magnitude, Mat &angle, float threshold)
{
  magnitude.create(src.size(), CV_32F);

  // Allocate temporary buffers
  Size size = src.size();
  Mat sobel_3dx; // per-channel horizontal derivative
  Mat sobel_3dy; // per-channel vertical derivative
  Mat sobel_dx(size, CV_32F);      // maximum horizontal derivative
  Mat sobel_dy(size, CV_32F);      // maximum vertical derivative
  Mat sobel_ag;  // final gradient orientation (unquantized)
  Mat smoothed;

  // Compute horizontal and vertical image derivatives on all color channels separately
  static const int KERNEL_SIZE = 7;
  // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
  GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

  short * ptrx  = (short *)sobel_3dx.data;
  short * ptry  = (short *)sobel_3dy.data;
  float * ptr0x = (float *)sobel_dx.data;
  float * ptr0y = (float *)sobel_dy.data;
  float * ptrmg = (float *)magnitude.data;

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
      int mag1 = ptrx[i+0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
      int mag2 = ptrx[i+1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
      int mag3 = ptrx[i+2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

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

static void selectScatteredFeatures(const std::vector<Linemod_embedding::Candidate> &candidates,
                                              std::vector<Linemod_embedding::element> &features, size_t num_features, float distance)
{
  features.clear();
  float distance_sq = distance * distance;
  int i = 0;
  while (features.size() < num_features)
  {
    auto c = candidates[i];

    // Add if sufficient distance away from any previously chosen feature
    bool keep = true;
    for (int j = 0; (j < (int)features.size()) && keep; ++j)
    {
      auto f = features[j];
      keep = (c.f.x - f.x)*(c.f.x - f.x) + (c.f.y - f.y)*(c.f.y - f.y) >= distance_sq;
    }
    if (keep)
      features.push_back(c.f);

    if (++i == (int)candidates.size())
    {
      // Start back at beginning, and relax required distance
      i = 0;
      distance -= 1.0f;
      distance_sq = distance * distance;
    }
  }
}

// Contains GRANULARITY and NORMAL_LUT
#include "normal_lut.i"

static void accumBilateral(long delta, long i, long j, long * A, long * b, int threshold)
{
  long f = std::abs(delta) < threshold ? 1 : 0;

  const long fi = f * i;
  const long fj = f * j;

  A[0] += fi * i;
  A[1] += fi * j;
  A[3] += fj * j;
  b[0]  += fi * delta;
  b[1]  += fj * delta;
}

static void quantizedNormals(const Mat& src, Mat& dst, int distance_threshold,
                      int difference_threshold)
{
  dst = Mat::zeros(src.size(), CV_8U);

  const unsigned short * lp_depth   = src.ptr<ushort>();
  unsigned char  * lp_normals = dst.ptr<uchar>();

  const int l_W = src.cols;
  const int l_H = src.rows;

  const int l_r = 5; // used to be 7
  const int l_offset0 = -l_r - l_r * l_W;
  const int l_offset1 =    0 - l_r * l_W;
  const int l_offset2 = +l_r - l_r * l_W;
  const int l_offset3 = -l_r;
  const int l_offset4 = +l_r;
  const int l_offset5 = -l_r + l_r * l_W;
  const int l_offset6 =    0 + l_r * l_W;
  const int l_offset7 = +l_r + l_r * l_W;

  const int l_offsetx = GRANULARITY / 2;
  const int l_offsety = GRANULARITY / 2;

  for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
  {
    const unsigned short * lp_line = lp_depth + (l_y * l_W + l_r);
    unsigned char * lp_norm = lp_normals + (l_y * l_W + l_r);

    for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
    {
      long l_d = lp_line[0];

      if (l_d < distance_threshold)
      {
        // accum
        long l_A[4]; l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
        long l_b[2]; l_b[0] = l_b[1] = 0;
        accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset1] - l_d,    0, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset3] - l_d, -l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset4] - l_d, +l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset6] - l_d,    0, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

        // solve
        long l_det =  l_A[0] * l_A[3] - l_A[1] * l_A[1];
        long l_ddx =  l_A[3] * l_b[0] - l_A[1] * l_b[1];
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

bool Linemod_feature::constructEmbedding(){
    {   // rgb embedding extract
        // Want features on the border to distinguish from background
        embedding.center_dep = depth.at<uint16_t>(depth.rows/2, depth.cols/2);
        Mat local_mask;
        if (!mask.empty())
        {
          erode(mask, local_mask, Mat(), Point(-1,-1), 1, BORDER_REPLICATE);
          subtract(mask, local_mask, local_mask);
        }
        bool no_mask = local_mask.empty();
        Mat magnitude, angle;
        quantizedOrientations(rgb, magnitude, angle, embedding.weak_threshold);
        embedding.angle = angle;
        std::vector<Linemod_embedding::Candidate> candidates;
        float threshold_sq = embedding.strong_threshold*embedding.strong_threshold;
        for (int r = 0; r < magnitude.rows; ++r)
        {
          const uchar* angle_r = angle.ptr<uchar>(r);
          const float* magnitude_r = magnitude.ptr<float>(r);
          const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);
          for (int c = 0; c < magnitude.cols; ++c)
          {
              if (no_mask || mask_r[c]){
                  uchar quantized = angle_r[c];
                  if (quantized > 0)
                  {
                    float score = magnitude_r[c];
                    if (score > threshold_sq)
                    {
                      candidates.push_back(Linemod_embedding::Candidate(c, r, getLabel(quantized), score));
                    }
                  }
              }

          }
        }
        // We require a certain number of features
        if (candidates.size() < embedding.num_features)
          return false;
        std::stable_sort(candidates.begin(), candidates.end());
        // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
        float distance = static_cast<float>(candidates.size() / embedding.num_features + 1);
        selectScatteredFeatures(candidates, embedding.rgb_embedding, embedding.num_features, distance);
//        auto bbox = cropTemplates(embedding.rgb_embedding);
    }
    {// depth embedding extract
        Mat normal;
        quantizedNormals(depth, normal, embedding.distance_threshold, embedding.difference_threshold);
        embedding.normal = normal;
//        cout << depth << endl;
        Mat local_mask;
        if (!mask.empty())
        {
          erode(mask, local_mask, Mat(), Point(-1,-1), 2, BORDER_REPLICATE);
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
        std::vector<Linemod_embedding::Candidate> candidates;
        bool no_mask = local_mask.empty();

        for (int r = 0; r < normal.rows; ++r)
        {
          const uchar* normal_r = normal.ptr<uchar>(r);
          const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

          for (int c = 0; c < normal.cols; ++c)
          {
            if (no_mask || mask_r[c])
            {
              uchar quantized = normal_r[c];

              if (quantized != 0 && quantized != 255) // background and shadow
              {
                int label = getLabel(quantized);

                float score = distances[label].at<float>(r, c);
                if (score >= embedding.extract_threshold)
                {
                  candidates.push_back( Linemod_embedding::Candidate(c, r, label, score));
                  ++label_counts[label];
                }
              }
            }
          }
        }
        // We require a certain number of features
        if (candidates.size() < embedding.num_features)
          return false;

        // Prefer large distances, but also want to collect features over all 8 labels.
        // So penalize labels with lots of candidates.
        for (size_t i = 0; i < candidates.size(); ++i)
        {
          auto& c = candidates[i];
          c.score /= (float)label_counts[c.f.label];
        }
        std::stable_sort(candidates.begin(), candidates.end());


        // Use heuristic based on object area for initial distance threshold
        float area = no_mask ? (float)normal.total() : (float)countNonZero(local_mask);
        float distance = sqrtf(area) / sqrtf((float)embedding.num_features) + 1.5f;
        selectScatteredFeatures(candidates, embedding.depth_embedding, embedding.num_features, distance);
//        cropTemplates(embedding.depth_embedding);
    }
    return true;
}
template <typename T>
static T clamp(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}
static uchar lutable[] = {4, 2, 1, 0, 0};
float Linemod_feature::similarity(Linemod_feature &other){
    int count = 0;
    float score = 0;
    auto& rgb_res = other.embedding.angle;
    int width = rgb_res.cols;
    int height = rgb_res.rows;
    for(auto element: embedding.rgb_embedding){
        if(other.embedding.center_dep>0 && embedding.center_dep>0){
            int normalize_x = element.x*embedding.center_dep/other.embedding.center_dep;
            int normalize_y = element.y*embedding.center_dep/other.embedding.center_dep;
            int z_1 = embedding.center_dep-depth.at<uint16_t>(element.y,element.x);
            int z_2 = other.embedding.center_dep-other.depth.at<uint16_t>(normalize_y,normalize_x);
            bool valid = std::abs(z_1-z_2) < embedding.z_check;
            if(valid){
                normalize_x = clamp(normalize_x, 0, width-1);
                normalize_y = clamp(normalize_y, 0, height-1);
                int ori = rgb_res.at<uchar>(normalize_y, normalize_x);
                ori = getLabel(ori);
                int diff = element.label - ori;
                diff = std::min(diff, 8-diff);
                score += lutable[diff];
                count++;
            }
        }
    }

    auto& dep_res = other.embedding.normal;
    width = dep_res.cols;
    height = dep_res.rows;
    for(auto element: embedding.depth_embedding){
        if(other.embedding.center_dep>0 && embedding.center_dep>0){
            int normalize_x = element.x*embedding.center_dep/other.embedding.center_dep;
            int normalize_y = element.y*embedding.center_dep/other.embedding.center_dep;
            int z_1 = embedding.center_dep-depth.at<uint16_t>(element.y,element.x);
            int z_2 = other.embedding.center_dep-other.depth.at<uint16_t>(normalize_y,normalize_x);
            bool valid = std::abs(z_1-z_2) < embedding.z_check;
            if(valid){
                normalize_x = clamp(normalize_x, 0, width-1);
                normalize_y = clamp(normalize_y, 0, height-1);
                int ori = dep_res.at<uchar>(normalize_y, normalize_x);
                ori = getLabel(ori);
                int diff = element.label - ori;
                diff = std::min(diff, 8-diff);
                score += lutable[diff];
                count++;
            }
        }
    }

    return score/count/4*100;
}

lchf::Linemod_feature Linemod_feature::write()
{
    lchf::Linemod_feature featrue_;
    if(!rgb.empty()){
        lchf::Mat_i_3* mat_i_3;
        Mat bgr[3];
        split(rgb,bgr);
        for(int i=0;i<3;i++){
            auto c = mat_i_3->add_channel();
            saveMat<uchar>(bgr[i], c);
        }
        featrue_.set_allocated_rgb(mat_i_3);
    }
    if(!depth.empty()){
        lchf::Mat_i* mat_i;
        saveMat<uint16_t>(depth, mat_i);
        featrue_.set_allocated_depth(mat_i);
    }
    if(!mask.empty()){
        lchf::Mat_i* mat_i;
        saveMat<uchar>(mask, mat_i);
        featrue_.set_allocated_mask(mat_i);
    }
    auto embedding_write = embedding.write();
    featrue_.set_allocated_embedding(&embedding_write);
}

lchf::Info Info::write()
{
    lchf::Info info_;
    if(!id.empty()){
        info_.set_id(id);
    }
    if(!t.empty()){
        lchf::Mat_f* mat_f;
        saveMat<float>(t, mat_f);
        info_.set_allocated_t(mat_f);
    }
    if(!R.empty()){
        lchf::Mat_f* mat_f;
        saveMat<float>(R, mat_f);
        info_.set_allocated_r(mat_f);
    }
}

void Info::read(lchf::Info &info_)
{
    id = info_.id();
    loadMat<float>(t, info_.t());
    loadMat<float>(R, info_.r());
}

lchf::Linemod_embedding Linemod_embedding::write()
{
    lchf::Linemod_embedding embedding_;
    if(!angle.empty()){
        lchf::Mat_i* mat_i;
        saveMat<uchar>(angle, mat_i);
        embedding_.set_allocated_angle(mat_i);
    }
    if(!normal.empty()){
        lchf::Mat_i* mat_i;
        saveMat<uchar>(normal, mat_i);
        embedding_.set_allocated_normal(mat_i);
    }
    embedding_.set_center_dep(center_dep);

    lchf::Linemod_embedding_ele_vector* lchf_ele_vec;
    for(int i=0;i<rgb_embedding.size();i++){
        auto& ele = rgb_embedding[i];
        auto lchf_ele = lchf_ele_vec->add_element();
        lchf_ele->set_x(ele.x);
        lchf_ele->set_y(ele.y);
        lchf_ele->set_label(ele.label);
    }
    embedding_.set_allocated_rgb_embedding(lchf_ele_vec);

    lchf::Linemod_embedding_ele_vector* lchf_ele_vec2;
    for(int i=0;i<depth_embedding.size();i++){
        auto& ele = depth_embedding[i];
        auto lchf_ele = lchf_ele_vec2->add_element();
        lchf_ele->set_x(ele.x);
        lchf_ele->set_y(ele.y);
        lchf_ele->set_label(ele.label);
    }
    embedding_.set_allocated_depth_embedding(lchf_ele_vec2);
}

void Linemod_embedding::read(lchf::Linemod_embedding &embedding_)
{
    center_dep = embedding_.center_dep();
    loadMat<uchar>(angle, embedding_.angle());
    loadMat<uchar>(normal, embedding_.normal());

    int rgb_ele_size = embedding_.rgb_embedding().element_size();
    rgb_embedding.resize(rgb_ele_size);
    for(int i=0; i<rgb_ele_size; i++){
        auto lchf_ele = embedding_.rgb_embedding().element(i);
        auto& ele = rgb_embedding[i];
        ele.x = lchf_ele.x();
        ele.y = lchf_ele.y();
        ele.label = lchf_ele.label();
    }

    int dep_ele_size = embedding_.depth_embedding().element_size();
    depth_embedding.resize(dep_ele_size);
    for(int i=0;i<dep_ele_size;i++){
        auto lchf_ele = embedding_.depth_embedding().element(i);
        auto& ele = depth_embedding[i];
        ele.x = lchf_ele.x();
        ele.y = lchf_ele.y();
        ele.label = lchf_ele.label();
    }
}

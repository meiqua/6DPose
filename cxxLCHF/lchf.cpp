// originally running linemod fails under py,
// this file is designed for running in c++ then binding to py
// however after some effort I find out that depth type matters
// now linemod works fine under py

#include "lchf.h"
#include <memory>
#include <chrono>
#include <assert.h>
using namespace std;
using namespace cv;

// for test
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(){
        double t = elapsed();
        std::cout << "elasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
int main(){
    Timer tmr;

    return 0;
}

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
static void orUnaligned8u(const uchar * src, const int src_stride,
                   uchar * dst, const int dst_stride,
                   const int width, const int height)
{
#if CV_SSE2
  volatile bool haveSSE2 = checkHardwareSupport(CPU_SSE2);
#if CV_SSE3
  volatile bool haveSSE3 = checkHardwareSupport(CPU_SSE3);
#endif
  bool src_aligned = reinterpret_cast<unsigned long long>(src) % 16 == 0;
#endif

  for (int r = 0; r < height; ++r)
  {
    int c = 0;

#if CV_SSE2
    // Use aligned loads if possible
    if (haveSSE2 && src_aligned)
    {
      for ( ; c < width - 15; c += 16)
      {
        const __m128i* src_ptr = reinterpret_cast<const __m128i*>(src + c);
        __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
        *dst_ptr = _mm_or_si128(*dst_ptr, *src_ptr);
      }
    }
#if CV_SSE3
    // Use LDDQU for fast unaligned load
    else if (haveSSE3)
    {
      for ( ; c < width - 15; c += 16)
      {
        __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src + c));
        __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
        *dst_ptr = _mm_or_si128(*dst_ptr, val);
      }
    }
#endif
    // Fall back to MOVDQU
    else if (haveSSE2)
    {
      for ( ; c < width - 15; c += 16)
      {
        __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
        __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
        *dst_ptr = _mm_or_si128(*dst_ptr, val);
      }
    }
#endif
    for ( ; c < width; ++c)
      dst[c] |= src[c];

    // Advance to next row
    src += src_stride;
    dst += dst_stride;
  }
}

static void spread(const Mat& src, Mat& dst, int T=5)
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
CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

/**
 * \brief Precompute response maps for a spread quantized image.
 *
 * Implements section 2.4 "Precomputing Response Maps."
 *
 * \param[in]  src           The source 8-bit spread quantized image.
 * \param[out] response_maps Vector of 8 response maps, one for each bit label.
 */
static void computeResponseMaps(const Mat& src, std::vector<Mat>& response_maps)
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
    const uchar* src_r = src.ptr(r);
    uchar* lsb4_r = lsb4.ptr(r);
    uchar* msb4_r = msb4.ptr(r);

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
    const __m128i* lut = reinterpret_cast<const __m128i*>(SIMILARITY_LUT);
    for (int ori = 0; ori < 8; ++ori)
    {
      __m128i* map_data = response_maps[ori].ptr<__m128i>();
      __m128i* lsb4_data = lsb4.ptr<__m128i>();
      __m128i* msb4_data = msb4.ptr<__m128i>();

      // Precompute the 2D response map S_i (section 2.4)
      for (int i = 0; i < (src.rows * src.cols) / 16; ++i)
      {
        // Using SSE shuffle for table lookup on 4 orientations at a time
        // The most/least significant 4 bits are used as the LUT index
        __m128i res1 = _mm_shuffle_epi8(lut[2*ori + 0], lsb4_data[i]);
        __m128i res2 = _mm_shuffle_epi8(lut[2*ori + 1], msb4_data[i]);

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
      uchar* map_data = response_maps[ori].ptr<uchar>();
      uchar* lsb4_data = lsb4.ptr<uchar>();
      uchar* msb4_data = msb4.ptr<uchar>();
      const uchar* lut_low = SIMILARITY_LUT + 32*ori;
      const uchar* lut_hi = lut_low + 16;

      for (int i = 0; i < src.rows * src.cols; ++i)
      {
        map_data[i] = std::max(lut_low[ lsb4_data[i] ], lut_hi[ msb4_data[i] ]);
      }
    }
  }
}


bool Linemod_feature::constructEmbedding(){
    {   // rgb embedding extract
        // Want features on the border to distinguish from background
        Mat local_mask;
        if (!mask.empty())
        {
          erode(mask, local_mask, Mat(), Point(-1,-1), 1, BORDER_REPLICATE);
          subtract(mask, local_mask, local_mask);
        }
        bool no_mask = local_mask.empty();
        Mat magnitude, angle;
        quantizedOrientations(rgb, magnitude, angle, embedding.weak_threshold);

        Mat rgb_spread;
        spread(angle, rgb_spread);
        computeResponseMaps(rgb_spread, embedding.rgb_res_map);

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
        cropTemplates(embedding.rgb_embedding);
    }
    {// depth embedding extract
        Mat normal;
        quantizedNormals(depth, normal, embedding.distance_threshold, embedding.difference_threshold);

        Mat depth_spread;
        spread(normal, depth_spread);
        computeResponseMaps(depth_spread, embedding.depth_res_map);

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

                // Accept if distance to a pixel belonging to a different label is greater than
                // some threshold. IOW, ideal feature is in the center of a large homogeneous
                // region.
                float score = distances[label].at<float>(r, c);
                if (score >= embedding.extract_threshold)
                {
                  candidates.push_back( Linemod_embedding::Candidate(c, r, label, score) );
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
        cropTemplates(embedding.depth_embedding);
    }
    return true;
}

float Linemod_feature::similarity(Linemod_feature &other){

}

#pragma once

#include <slimage/image.hpp>
#include <slimage/error.hpp>
#include <slimage/algorithm.hpp>
#include <opencv2/highgui/highgui.hpp>
#define SLIMAGE_OPENCV_INC
#include <functional>

namespace slimage
{

	namespace detail
	{
		template<typename K, unsigned CC>
		struct OpenCvImageType;

		#define SLIMAGE_OPENCV_IMG_TYPE(K,CC,CVT) \
			template<> struct OpenCvImageType<K,CC> { \
				static constexpr int value = CVT; \
			};

		#define SLIMAGE_OPENCV_IMG_TYPE_BATCH(K,CVT) \
			SLIMAGE_OPENCV_IMG_TYPE(K,1,CVT##C1) \
			SLIMAGE_OPENCV_IMG_TYPE(K,3,CVT##C3) \
			SLIMAGE_OPENCV_IMG_TYPE(K,4,CVT##C4)

		SLIMAGE_OPENCV_IMG_TYPE_BATCH(char, CV_8S)
		SLIMAGE_OPENCV_IMG_TYPE_BATCH(unsigned char, CV_8U)
		SLIMAGE_OPENCV_IMG_TYPE_BATCH(uint16_t, CV_16U)
		SLIMAGE_OPENCV_IMG_TYPE_BATCH(int, CV_32S)
		SLIMAGE_OPENCV_IMG_TYPE_BATCH(float, CV_32F)
		SLIMAGE_OPENCV_IMG_TYPE_BATCH(double, CV_64F)

		#undef SLIMAGE_OPENCV_IMG_TYPE_BATCH
		#undef SLIMAGE_OPENCV_IMG_TYPE

		template<typename K, unsigned CC>
		struct OpenCvCopyPixelsImpl;

		template<typename K>
		struct OpenCvCopyPixelsImpl<K,1>
		{
			static void function(const K* src, const K* src_end, K* dst)
			{ std::copy(src, src_end, dst); }
		};

		template<typename K>
		struct OpenCvCopyPixelsImpl<K,3>
		{
			static void function(const K* src, const K* src_end, K* dst)
			{ Copy_RGB_to_BGR(src, src_end, dst); }
		};

		template<typename K>
		struct OpenCvCopyPixelsImpl<K,4>
		{
			static void function(const K* src, const K* src_end, K* dst)
			{ Copy_RGBA_to_BGRA(src, src_end, dst); }
		};

	}

	/** Converts a typed slimage image to an OpenCV image */
	template<typename K, unsigned CC>
	cv::Mat ConvertToOpenCv(const Image<K,CC>& img)
	{
	 	cv::Mat mat(img.height(), img.width(), detail::OpenCvImageType<K,CC>::value);
		CopyScanlines(
			img,
			[&mat](unsigned y) { return mat.ptr<K>(y,0); },
			detail::OpenCvCopyPixelsImpl<K,CC>::function);
	 	return mat;
	}

	/** Converts an anonymous slimage image to an OpenCV image */
	inline
	cv::Mat ConvertToOpenCv(const AnonymousImage& aimg)
	{
		#define SLIMAGE_ConvertToOpenCv_HELPER(K,CC) \
			if(anonymous_is<K,CC>(aimg)) return ConvertToOpenCv(anonymous_cast<K,CC>(aimg));

		#define SLIMAGE_ConvertToOpenCv_HELPER_BATCH(K) \
			SLIMAGE_ConvertToOpenCv_HELPER(K,1) \
			SLIMAGE_ConvertToOpenCv_HELPER(K,3) \
			SLIMAGE_ConvertToOpenCv_HELPER(K,4)
		
		SLIMAGE_ConvertToOpenCv_HELPER_BATCH(char)
		SLIMAGE_ConvertToOpenCv_HELPER_BATCH(unsigned char)
		SLIMAGE_ConvertToOpenCv_HELPER_BATCH(uint16_t)
		SLIMAGE_ConvertToOpenCv_HELPER_BATCH(int)
		SLIMAGE_ConvertToOpenCv_HELPER_BATCH(float)
		SLIMAGE_ConvertToOpenCv_HELPER_BATCH(double)
		throw ConversionException("Unknown type of AnonymousImage in ConvertToOpenCv");
		
		#undef SLIMAGE_ConvertToOpenCv_HELPER_BATCH
		#undef SLIMAGE_ConvertToOpenCv_HELPER
	}

	/** Converts an OpenCV image to a typed slimage image */
	template<typename K, unsigned CC>
	Image<K,CC> ConvertToSlimage(const cv::Mat& mat)
	{
		#define TOSTRING(X) #X
		if(mat.type() != detail::OpenCvImageType<K,CC>::value)
			throw ConversionException("cv::Mat does not have expected type: element_type=" TOSTRING(K) ", channel count=" TOSTRING(CC));
		#undef TOSTRING
		Image<K,CC> img(mat.cols, mat.rows);
		CopyScanlines(
			[&mat](unsigned y) { return mat.ptr<K>(y,0); },
			img,
			detail::OpenCvCopyPixelsImpl<K,CC>::function);
		return img;
	}

	/** Converts an OpenCV image to an anonymous slimage image */
	inline
	AnonymousImage ConvertToSlimage(const cv::Mat& mat)
	{
		#define SLIMAGE_ConvertToSlimage_HELPER(K,CC,CVT) \
			if(mat.type() == CVT) return make_anonymous(ConvertToSlimage<K,CC>(mat));

		#define SLIMAGE_ConvertToSlimage_HELPER_BATCH(K,CVT) \
			SLIMAGE_ConvertToSlimage_HELPER(K,1,CVT##C1) \
			SLIMAGE_ConvertToSlimage_HELPER(K,3,CVT##C3) \
			SLIMAGE_ConvertToSlimage_HELPER(K,4,CVT##C4)
		
		SLIMAGE_ConvertToSlimage_HELPER_BATCH(char, CV_8S)
		SLIMAGE_ConvertToSlimage_HELPER_BATCH(unsigned char, CV_8U)
		SLIMAGE_ConvertToSlimage_HELPER_BATCH(uint16_t, CV_16U)
		SLIMAGE_ConvertToSlimage_HELPER_BATCH(int, CV_32S)
		SLIMAGE_ConvertToSlimage_HELPER_BATCH(float, CV_32F)
		SLIMAGE_ConvertToSlimage_HELPER_BATCH(double, CV_64F)
		throw ConversionException("Unknown type of cv::Mat in ConvertToSlimage(cv::Mat)");
		
		#undef SLIMAGE_ConvertToSlimage_HELPER_BATCH
		#undef SLIMAGE_ConvertToSlimage_HELPER
	}

	/** Saves an image to a file using OpenCV */
	inline
	void OpenCvSave(const std::string& filename, const AnonymousImage& img)
	{
		cv::imwrite(filename, ConvertToOpenCv(img)); // TODO correct error handling?
	}

	/** Loads an image from a file using OpenCV */
	inline
	AnonymousImage OpenCvLoad(const std::string& filename)
	{
		cv::Mat mat = cv::imread(filename); // TODO correct error handling?
		if(mat.empty()) {
			throw IoException(filename, "Empty image (does the file exists?)");
		}
		return ConvertToSlimage(mat);
	}

}

#pragma once

#ifndef SLIMAGE_OPENCV_INC
#  include <slimage/opencv.hpp>
#  define SLIMAGE_OPENCV_INC_UNDO
#endif
#include <slimage/algorithm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

namespace slimage
{
	inline
	void GuiWait(unsigned int delay=0)
	{
		if(delay == 0) {
			// wait for any key press
			while(cv::waitKey(20) == -1);
		}
		else {
			// wait for time
			cv::waitKey(delay);
		}
	}

	template<typename IMAGE>
	void GuiShow(const std::string& caption, const IMAGE& img, unsigned int delay=3)
	{
		cv::imshow(caption.c_str(), ConvertToOpenCv(img));
		cv::waitKey(delay); // need to wait a bit, otherwise image is not displayed properly
	}

	inline
	void GuiShow(const std::string& caption, const slimage::Image1f& img, unsigned int delay=3)
	{
		GuiShow(caption, Convert(img, [](float v) { return static_cast<unsigned char>(255.0f*v); }), delay);
	}

}

#ifdef SLIMAGE_OPENCV_INC_UNDO
#  undef SLIMAGE_OPENCV_INC
#endif

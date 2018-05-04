#pragma once

#include <slimage/image.hpp>
#include <slimage/error.hpp>
#include <slimage/algorithm.hpp>
#include <QtGui/QImage>
#define SLIMAGE_QT_INC
#include <algorithm>

namespace slimage
{

	inline
	QImage ConvertToQt(const Image1ub& mask)
	{
		unsigned int h = mask.height();
		unsigned int w = mask.width();
		QImage imgQt(w, h, QImage::Format_Indexed8);
		QVector<QRgb> colors(256);
		for(unsigned int i=0; i<=255; i++) {
			colors[i] = qRgb(i,i,i);
		}
		imgQt.setColorTable(colors);
		for(uint i=0; i<h; i++) {
			const unsigned char* src = mask.pixel_pointer(0, i);
			unsigned char* dst = imgQt.scanLine(i);
			std::copy(src, src + w, dst);
		}
		return imgQt;
	}

	inline
	QImage ConvertToQt(const Image3ub& img)
	{
		unsigned int h = img.height();
		unsigned int w = img.width();
		QImage imgQt(w, h, QImage::Format_RGB32);
		for(unsigned int i=0; i<h; i++) {
			const unsigned char* src = img.pixel_pointer(0, i);
			unsigned char* dst = imgQt.scanLine(i);
			Copy_RGB_to_BGRA(src, src + 3*w, dst, static_cast<unsigned char>(255));
		}
		return imgQt;
	}

	inline
	QImage ConvertToQt(const Image4ub& img)
	{
		unsigned int h = img.height();
		unsigned int w = img.width();
		QImage imgQt(w, h, QImage::Format_ARGB32);
		for(unsigned int i=0; i<h; i++) {
			const unsigned char* src = img.pixel_pointer(0, i);
			unsigned char* dst = imgQt.scanLine(i);
			Copy_RGBA_to_BGRA(src, src + 4*w, dst);
		}
		return imgQt;
	}


	inline
	QImage ConvertToQt(const AnonymousImage& aimg)
	{
		if(anonymous_is<unsigned char,1>(aimg)) {
			return ConvertToQt(anonymous_cast<unsigned char, 1>(aimg));
		}
		if(anonymous_is<unsigned char, 3>(aimg)) {
			return ConvertToQt(anonymous_cast<unsigned char, 3>(aimg));
		}
		if(anonymous_is<unsigned char, 4>(aimg)) {
			return ConvertToQt(anonymous_cast<unsigned char, 4>(aimg));
		}
		throw ConversionException("Invalid type of AnonymousImage for ConvertToQt");
	}

	inline
	AnonymousImage ConvertToSlimage(const QImage& qimg)
	{
//		std::cout << qimg.format() << std::endl;
		if(qimg.format() == QImage::Format_Indexed8) {
			unsigned int w = qimg.width();
			unsigned int h = qimg.height();
			Image1ub img(w, h);
			for(unsigned int i=0; i<h; i++) {
				const unsigned char* src = qimg.scanLine(i);
				unsigned char* dst = img.pixel_pointer(0, i);
				std::copy(src, src+w, dst);
			}
			return make_anonymous(img);
		}
		if(qimg.format() == QImage::Format_RGB32) {
			unsigned int h = qimg.height();
			unsigned int w = qimg.width();
			Image3ub img(w, h);
			for(unsigned int i=0; i<h; i++) {
				const unsigned char* src = qimg.scanLine(i);
				unsigned char* dst = img.pixel_pointer(0, i);
				Copy_RGBA_to_BGR(src, src + 4*w, dst);
			}
			return make_anonymous(img);
		}
		if(qimg.format() == QImage::Format_ARGB32) {
			unsigned int h = qimg.height();
			unsigned int w = qimg.width();
			Image4ub img(w, h);
			for(unsigned int i=0; i<h; i++) {
				const unsigned char* src = qimg.scanLine(i);
				unsigned char* dst = img.pixel_pointer(0, i);
				Copy_RGBA_to_BGRA(src, src + 4*w, dst);
			}
			return make_anonymous(img);
		}
		throw new ConversionException("Invalid type of QImage for ConvertToSlimage(QImage)");
	}

	inline
	void QtSave(const std::string& filename, const AnonymousImage& img)
	{ ConvertToQt(img).save(QString::fromStdString(filename)); }

	inline
	AnonymousImage QtLoad(const std::string& filename)
	{ return ConvertToSlimage(QImage(QString::fromStdString(filename))); }

}

#pragma once

#include <string>
#include <stdexcept>

namespace slimage
{

	struct IoException
	: public std::runtime_error
	{
	public:
		IoException(const std::string& filename, const std::string& msg)
		: std::runtime_error("slimage::IoException regarding file '" + filename + "': " + msg) {}
	};

	struct ConversionException
	: public std::runtime_error
	{
	public:
		ConversionException(const std::string& msg)
		: std::runtime_error("slimage::ConversionException: " + msg) {}
	};

	struct CastException
	: public std::runtime_error
	{
	public:
		CastException()
		: std::runtime_error("slimage::CastException: Invalid anonymous_cast<> -- source image is not of specified type!") {}
	};



}
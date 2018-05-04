#pragma once

#include <slimage/pixel.hpp>
#include <slimage/iterator.hpp>
#include <slimage/error.hpp>
#include <algorithm>
#include <tuple>
#include <vector>
#include <memory>
#include <cassert>
#include <stdint.h>

namespace slimage
{
	template<typename K, unsigned CC, typename IDX=unsigned>
	class Image
	{
	public:
		using idx_t = IDX;
		using element_t = K;
		using iterator_t = Iterator<K,CC>;
		using const_iterator_t = Iterator<const K,CC>;
		using reference_t = typename iterator_t::reference;
		using const_reference_t = typename const_iterator_t::reference;
		using dim_t = std::tuple<unsigned,unsigned>;

		Image()
		:	width_(0),
			height_(0)
		{}

		Image(idx_t width, idx_t height)
		:	width_(width),
			height_(height),
			data_(CC*width*height)
		{}

		Image(idx_t width, idx_t height, const Pixel<K,CC>& value)
		:	width_(width),
			height_(height),
			data_(CC*width*height)
		{
			std::fill(begin(), end(), value);
		}

		Image(dim_t dim)
		:	Image(std::get<0>(dim), std::get<1>(dim))
		{}

		Image(dim_t dim, const Pixel<K,CC>& value)
		:	Image(std::get<0>(dim), std::get<1>(dim), value)
		{}

		void resize(idx_t width, idx_t height)
		{
			width_ = width;
			height_ = height;
			data_.resize(CC*width_*height_);
		}

		void resize(dim_t dim)
		{ resize(std::get<0>(dim), std::get<1>(dim)); }

		bool empty() const
		{ return width_ == 0 && height_ == 0; }

		/** Width of image */
		unsigned width() const
		{ return width_; }

		/** Height of image */
		unsigned height() const
		{ return height_; }

		dim_t dimensions() const
		{ return std::make_tuple(width(), height()); }

		/** Number of elements per pixel */
		unsigned channelCount() const
		{ return CC; }

		/** Number of pixels, i.e. width()*height() */
		size_t size() const
		{ return width_*height_; }

		/** Number of elements in the whole image, i.e. width()*height()*channelCount() */
		size_t numElementsImage() const
		{ return data_.size(); }

		/** Number of elements in a line, i.e. width()*channelCount() */
		size_t numElementsScanline() const
		{ return CC*width_; }

		reference_t operator[](idx_t i)
		//{ return make_ref(pixel_pointer(i), Integer<CC>()); }
		{ return *(begin() + i); }

		const_reference_t operator[](idx_t i) const
		//{ return make_ref(pixel_pointer(i), Integer<CC>()); }
		{ return *(begin() + i); }

		reference_t operator()(idx_t x, idx_t y)
		//{ return make_ref(pixel_pointer(x,y), Integer<CC>()); }
		{ return *(begin() + index(x,y)); }

		const_reference_t operator()(idx_t x, idx_t y) const
		//{ return make_ref(pixel_pointer(x,y), Integer<CC>()); }
		{ return *(begin() + index(x,y)); }

		iterator_t begin()
		{ return iterator_t{pixel_pointer()}; }

		iterator_t end()
		{ return iterator_t{pixel_pointer() + CC*size()}; }

		const_iterator_t begin() const
		{ return const_iterator_t{pixel_pointer()}; }

		const_iterator_t end() const
		{ return const_iterator_t{pixel_pointer() + CC*size()}; }

		bool isValidIndex(idx_t x, idx_t y) const
		{ return 0 <= x && x < width_ && 0 <= y && y < height_; }

		size_t index(idx_t x, idx_t y) const
		{
			assert(isValidIndex(x,y));
			return x + y*width_;
		}

		element_t* pixel_pointer(idx_t x, idx_t y)
		{ return data_.data() + CC*index(x,y); } 

		const element_t* pixel_pointer(idx_t x, idx_t y) const
		{ return data_.data() + CC*index(x,y); } 

		element_t* pixel_pointer(size_t i=0)
		{
			assert(i < size()); // TODO this is a bit of a hack but we need to support end()
			return data_.data() + CC*i;
		} 

		const element_t* pixel_pointer(size_t i=0) const
		{
			assert(i < size()); // TODO this is a bit of a hack but we need to support end()
			return data_.data() + CC*i;
		} 

	private:
		idx_t width_, height_;
		std::vector<element_t> data_;
	};

	#define SLIMAGE_CREATE_TYPEDEF(K,CC,S)\
		typedef Pixel<K,CC> Pixel##CC##S; \
		typedef Image<K,CC> Image##CC##S;

	SLIMAGE_CREATE_TYPEDEF(unsigned char, 1, ub)
	SLIMAGE_CREATE_TYPEDEF(unsigned char, 3, ub)
	SLIMAGE_CREATE_TYPEDEF(unsigned char, 4, ub)
	SLIMAGE_CREATE_TYPEDEF(float, 1, f)
	SLIMAGE_CREATE_TYPEDEF(float, 2, f)
	SLIMAGE_CREATE_TYPEDEF(float, 3, f)
	SLIMAGE_CREATE_TYPEDEF(float, 4, f)
	SLIMAGE_CREATE_TYPEDEF(double, 1, d)
	SLIMAGE_CREATE_TYPEDEF(double, 2, d)
	SLIMAGE_CREATE_TYPEDEF(double, 3, d)
	SLIMAGE_CREATE_TYPEDEF(double, 4, d)
	SLIMAGE_CREATE_TYPEDEF(uint16_t, 1, ui16)
	SLIMAGE_CREATE_TYPEDEF(int, 1, i)

	namespace detail
	{
		struct AnonymousInterface
		{
			virtual ~AnonymousInterface() {}
			virtual unsigned width() const = 0;
			virtual unsigned height() const = 0;
			virtual unsigned channelCount() const = 0;
		};

		template<typename K, unsigned CC>
		struct AnonymousImpl
		:	public AnonymousInterface
		{
			AnonymousImpl(const Image<K,CC>& img)
			:	img(img)
			{}

			AnonymousImpl(Image<K,CC>&& img)
			:	img(std::forward(img))
			{}

			unsigned width() const
			{ return img.width(); }

			unsigned height() const
			{ return img.height(); }

			unsigned channelCount() const
			{ return img.channelCount(); }

			Image<K,CC> img;
		};
	}

	using AnonymousImage = std::shared_ptr<detail::AnonymousInterface>;

	template<typename K, unsigned CC>
	bool anonymous_is(const AnonymousImage& aimg)
	{ return static_cast<bool>(std::dynamic_pointer_cast<detail::AnonymousImpl<K,CC>>(aimg)); }

	template<typename K, unsigned CC>
	Image<K,CC> anonymous_cast(const AnonymousImage& aimg)
	{
		auto p = std::dynamic_pointer_cast<detail::AnonymousImpl<K,CC>>(aimg);
		if(!p) {
			throw CastException();
		}
		return p->img;
	}

	template<typename K, unsigned CC>
	AnonymousImage make_anonymous(const Image<K,CC>& img)
	{ return std::make_shared<detail::AnonymousImpl<K,CC>>(img); }

	template<typename K, unsigned CC>
	AnonymousImage make_anonymous(Image<K,CC>&& img)
	{ return std::make_shared<detail::AnonymousImpl<K,CC>>(img); }

}

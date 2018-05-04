#pragma once

#include <array>
#include <type_traits>
#include <cassert>

namespace slimage
{

	namespace detail
	{
		template<typename K, unsigned CC>
		struct PixelType
		{
			using type = std::array<K,CC>;
		};

		template<typename K>
		struct PixelType<K,1>
		{
			using type = K;
		};
	}

	template<typename K, unsigned CC>
	using Pixel = typename detail::PixelType<K,CC>::type;

	template<typename K, unsigned CC>
	class PixelReference
	{
	public:
		using element_t = K;
		using pointer_t = element_t*;
		using reference_t = element_t&;
		using base_t = typename std::remove_const<K>::type;
		using pixel_t = Pixel<base_t,CC>;

		PixelReference(pointer_t ptr)
		:	ptr_(ptr)
		{}

		void operator=(const pixel_t& p)
		{ std::copy(p.begin(), p.end(), ptr_); }

		operator pixel_t() const
		{
			pixel_t result;
			std::copy(ptr_, ptr_+CC, result.begin());
			return result;
		}

		reference_t operator[](unsigned i) const
		{ assert(i < CC); return ptr_[i]; }

	private:
		pointer_t ptr_;
	};

	template<unsigned CC>
	struct Integer {};

	template<typename K, unsigned CC>
	PixelReference<K,CC> make_ref(K* p, Integer<CC>)
	{ return PixelReference<K,CC>(p); }

	template<typename K>
	K& make_ref(K* p, Integer<1>)
	{ return *p; }

	template<typename K, unsigned CC>
	struct PixelTraits
	{
		using pixel_t = Pixel<K,CC>;
		using pointer_t = Pixel<K,CC>*;
		using reference_t = PixelReference<K,CC>;
	};

	template<typename K>
	struct PixelTraits<K,1>
	{
		using pixel_t = K;
		using pointer_t = K*;
		using reference_t = K&;
	};
}

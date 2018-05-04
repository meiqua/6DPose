#pragma once

#include <slimage/pixel.hpp>
#include <vector>
#include <iterator>
#include <cassert>

namespace slimage
{

	namespace detail
	{
		template<typename K>
		K* post_add(K*& p, size_t n)
		{
			K* result = p;
			p += n;
			return result;
		}

		template<typename K>
		K* post_sub(K*& p, size_t n)
		{
			K* result = p;
			p -= n;
			return result;
		}
	}

	template<typename K, unsigned CC>
	class Iterator
	:	public std::iterator<
			std::random_access_iterator_tag,
			typename PixelTraits<K,CC>::pixel_t,
			std::ptrdiff_t,
			typename PixelTraits<K,CC>::pointer_t,
			typename PixelTraits<K,CC>::reference_t
		>
	{
	public:
		using element_t = K;
		using std_iterator_base_t = std::iterator<
			std::random_access_iterator_tag,
			typename PixelTraits<K,CC>::pixel_t,
			std::ptrdiff_t,
			typename PixelTraits<K,CC>::pointer_t,
			typename PixelTraits<K,CC>::reference_t
		>;
		using reference = typename std_iterator_base_t::reference;
		using difference_type = typename std_iterator_base_t::difference_type;

		Iterator(element_t* ptr = nullptr)
		:	ptr_(ptr)
		{}

		Iterator(const Iterator& it) = default;
		Iterator& operator=(const Iterator& it) = default;

		reference operator*() const
		{ return make_ref(ptr_, Integer<CC>()); }

		reference operator[](difference_type n) const
		{ return make_ref(ptr_ + CC*n, Integer<CC>()); }

		element_t* operator->() const
		{
			assert(CC == 1); // FIXME
			return ptr_;
		}

		Iterator& operator++()
		{ ptr_ += CC; return *this; }

		Iterator operator++(int)
		{ return Iterator(detail::post_add(ptr_,CC)); }

		Iterator& operator+=(difference_type n)
		{ ptr_ += CC*n; return *this; }

		Iterator operator+(difference_type n) const
		{ return Iterator(ptr_ + CC*n); }

		Iterator& operator--()
		{ ptr_ -= CC; return *this; }

		Iterator operator--(int)
		{ return Iterator(detail::post_sub(ptr_,CC)); }

		Iterator& operator-=(difference_type n)
		{ ptr_ -= CC*n; return *this; }

		Iterator operator-(difference_type n) const
		{ return Iterator(ptr_ - CC*n); }

		element_t* base() const
		{ return ptr_; }

	private:
		element_t* ptr_;
	};

	template<typename K1, typename K2, unsigned CC>
	bool operator==(const Iterator<K1,CC>& a, const Iterator<K2,CC>& b)
	{ return a.base() == b.base(); }

	template<typename K1, typename K2, unsigned CC>
	bool operator!=(const Iterator<K1,CC>& a, const Iterator<K2,CC>& b)
	{ return a.base() != b.base(); }

	template<typename K1, typename K2, unsigned CC>
	bool operator<(const Iterator<K1,CC>& a, const Iterator<K2,CC>& b)
	{ return a.base() < b.base(); }

	template<typename K1, typename K2, unsigned CC>
	bool operator<=(const Iterator<K1,CC>& a, const Iterator<K2,CC>& b)
	{ return a.base() <= b.base(); }

	template<typename K1, typename K2, unsigned CC>
	bool operator>(const Iterator<K1,CC>& a, const Iterator<K2,CC>& b)
	{ return a.base() > b.base(); }

	template<typename K1, typename K2, unsigned CC>
	bool operator>=(const Iterator<K1,CC>& a, const Iterator<K2,CC>& b)
	{ return a.base() >= b.base(); }

	template<typename K1, typename K2, unsigned CC>
	auto operator-(const Iterator<K1,CC>& a, const Iterator<K2,CC>& b) -> decltype(a.base() - b.base())
	{ return a.base() - b.base(); }

}

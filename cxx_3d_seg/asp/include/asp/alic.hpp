#pragma once

#include <asp/pds.hpp>
#include <asp/segmentation.hpp>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>

namespace asp {

namespace detail
{
	inline
	std::tuple<int,int> GetRange(int a, int b, float x, float w)
	{
		return std::make_tuple(
			std::max<int>(a, std::floor(x - w)),
			std::min<int>(b, std::ceil(x + w))
		);
	}

	/** Compute superpixel radius from density */
	inline
	float DensityToRadius(float density)
	{ return std::sqrt(1.0f / (density*3.1415f)); } // rho = 1 / (r*r*pi) => r = sqrt(rho/pi)

	/** Accumulate pixel data into superpixels */
	template<typename T>
	struct SegmentAccumulator
	{
		using acc_t = typename SegmentBase<T>::accumulate_t;

		SegmentAccumulator()
		:	sum_(acc_t::Zero())
		{}

		void add(const SegmentBase<T>& v)
		{ sum_.accumulate(v); }

		bool empty() const
		{ return sum_.num == 0.0f; }

		SegmentBase<T> mean() const
		{
			if(empty()) {
				return sum_;
			}
			auto seg = sum_;
			seg.normalize(sum_.num);
			seg.num = sum_.num; // preserve accumulated weight
			return seg;
		}

	private:
		acc_t sum_;
	};

}


/** Adaptive Local Iterative Clustering superpixel algorithm */
template<typename T, typename F>
Segmentation<T> ALIC(const slimage::Image<Pixel<T>,1>& input, const std::vector<Seed>& seeds, F dist)
{
	constexpr unsigned ITERATIONS = 5;
	constexpr float LAMBDA = 3.0f;
	const unsigned width = input.width();
	const unsigned height = input.height();
	// initialize
	Segmentation<T> s;
	s.input = input;
	s.superpixels.resize(seeds.size());
	for(size_t i=0; i<seeds.size(); i++) {
		const Seed& seed = seeds[i];
		auto& sp = s.superpixels[i];
		reinterpret_cast<SegmentBase<T>&>(sp) = reinterpret_cast<const SegmentBase<T>&>(
			input(std::floor(seed.position.x()), std::floor(seed.position.y())));
		sp.num = 1.0f;
		sp.position = seed.position;
		sp.density = seed.density;
		sp.radius = detail::DensityToRadius(sp.density);
	}
	s.indices = slimage::Image<int,1>{width, height};
	s.weights = slimage::Image1f{width, height};
	// iterate
	for(unsigned k=0; k<ITERATIONS; k++) {
		// reset weights
		std::fill(s.indices.begin(), s.indices.end(), -1);
		std::fill(s.weights.begin(), s.weights.end(), std::numeric_limits<float>::max());
		// iterate over all superpixels
		for(size_t sid=0; sid<s.superpixels.size(); sid++) {
			const auto& sp = s.superpixels[sid];
			// compute superpixel bounding box
			int x1, x2, y1, y2;
			std::tie(x1,x2) = detail::GetRange(0,  width, sp.position.x(), LAMBDA*sp.radius);
			std::tie(y1,y2) = detail::GetRange(0, height, sp.position.y(), LAMBDA*sp.radius);
			// iterate over superpixel bounding box
			for(int y=y1; y<y2; y++) {
				for(int x=x1; x<x2; x++) {
					const auto& val = input(x,y);
					if(!val.valid()) {
						continue;
					}
					float d = dist(sp, val);
					if(d < s.weights(x,y)) {
						s.weights(x,y) = d;
						s.indices(x,y) = sid;
					}
				}
			}
		}
		// update superpixels
		std::vector<detail::SegmentAccumulator<T>> acc(s.superpixels.size(), detail::SegmentAccumulator<T>{});
		for(unsigned y=0; y<s.indices.height(); y++) {
			for(unsigned x=0; x<s.indices.width(); x++) {
				int sid = s.indices(x,y);
				if(sid >= 0) {
					acc[sid].add(input(x,y));
				}
			}
		}
		for(size_t i=0; i<s.superpixels.size(); i++) {
			auto& sp = s.superpixels[i];
			reinterpret_cast<SegmentBase<T>&>(sp) = acc[i].mean();
			sp.radius = detail::DensityToRadius(sp.density);
		}
	}
	return s;
}


}

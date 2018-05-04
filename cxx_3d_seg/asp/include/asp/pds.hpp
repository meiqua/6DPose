#pragma once

#include <asp/segmentation.hpp>
#include <Eigen/Dense>
#include <vector>

namespace asp {

enum class PoissonDiskSamplingMethod
{
	Random,
	Grid,
	FloydSteinberg,
	FloydSteinbergExpo
};

std::vector<Eigen::Vector2f> PoissonDiskSampling(PoissonDiskSamplingMethod method, const Eigen::MatrixXf& density);

/** Superpixel seed */
struct Seed
{
	Eigen::Vector2f position;
	float density;
};

/** Compute seeds accordingly to pixel density values */
template<typename T>
std::vector<Seed> ComputeSeeds(PoissonDiskSamplingMethod method, const slimage::Image<Pixel<T>,1>& input)
{
	const unsigned width = input.width();
	const unsigned height = input.height();
	Eigen::MatrixXf density{width, height};
	for(unsigned y=0; y<height; y++) {
		for(unsigned x=0; x<width; x++) {
			density(x,y) = ((Pixel<T>&)input(x,y)).density;
		}
	}
	std::vector<Eigen::Vector2f> pntseeds = PoissonDiskSampling(method, density);
	std::vector<Seed> seeds(pntseeds.size());
	for(unsigned i=0; i<pntseeds.size(); i++) {
		auto& sp = seeds[i];
		sp.position = pntseeds[i];
		const Pixel<T>& inp_px = input(std::floor(sp.position.x()), std::floor(sp.position.y()));
		sp.density = inp_px.density; // FIXME use bb?
	}
	return seeds;
}

}

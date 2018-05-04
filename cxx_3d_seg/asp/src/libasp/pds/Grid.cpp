#include <Eigen/Dense>
#include <vector>
#include <random>

namespace asp
{

std::vector<Eigen::Vector2f> PdsRandom(const Eigen::MatrixXf& density)
{
	std::mt19937 rnd_engine; // FIXME seed?
	std::uniform_real_distribution<float> unif(0.0f, 1.0f);
	std::vector<Eigen::Vector2f> seeds;
	for(unsigned int iy=0; iy<density.cols(); iy++) {
		for(unsigned int ix=0; ix<density.rows(); ix++) {
			if(unif(rnd_engine) < density(ix,iy))
				seeds.push_back(Eigen::Vector2f(ix, iy));
		}
	}
	return seeds;
}

std::vector<Eigen::Vector2f> PdsGrid(const Eigen::MatrixXf& density)
{
	const float width = static_cast<float>(density.rows());
	const float height = static_cast<float>(density.cols());
	const float numf = density.sum();
	const float d = std::sqrt(float(width*height) / numf);
	const unsigned int Nx = static_cast<unsigned int>(std::ceil(width / d));
	const unsigned int Ny = static_cast<unsigned int>(std::ceil(height / d));
	const float Dx = width / static_cast<float>(Nx);
	const float Dy = height / static_cast<float>(Ny);
	const float Hx = Dx/2.0f;
	const float Hy = Dy/2.0f;

	std::vector<Eigen::Vector2f> seeds;
	seeds.reserve(Nx*Ny);
	for(unsigned int iy=0; iy<Ny; iy++) {
		float y = Hy + Dy * static_cast<float>(iy);
		for(unsigned int ix=0; ix<Nx; ix++) {
			float x = Hx + Dx * static_cast<float>(ix);
			seeds.push_back(Eigen::Vector2f(x, y));
		}
	}

	return seeds;
}

}

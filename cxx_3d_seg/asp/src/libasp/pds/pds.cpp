#include <asp/pds.hpp>

namespace asp
{

std::vector<Eigen::Vector2f> PdsRandom(const Eigen::MatrixXf& density_inp);
std::vector<Eigen::Vector2f> PdsGrid(const Eigen::MatrixXf& density_inp);
std::vector<Eigen::Vector2f> PdsFloydSteinberg(const Eigen::MatrixXf& density_inp);
std::vector<Eigen::Vector2f> PdsFloydSteinbergExpo(const Eigen::MatrixXf& density_inp);

std::vector<Eigen::Vector2f> PoissonDiskSampling(PoissonDiskSamplingMethod method, const Eigen::MatrixXf& density)
{
	#define OPT(Q) case PoissonDiskSamplingMethod::Q: return Pds##Q(density);
	switch(method) {
		OPT(Random)
		OPT(Grid)
		OPT(FloydSteinberg)
		OPT(FloydSteinbergExpo)
		default: return {};
	}
}

}

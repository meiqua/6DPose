#include <slimage/algorithm.hpp>
#include <asp/algos.hpp>
#include <asp/alic.hpp>

namespace asp
{

	Segmentation<PixelRgb> SuperpixelsAsp(const slimage::Image3ub& color, const slimage::Image1f& density, const AspParameters& opt)
	{
		constexpr PoissonDiskSamplingMethod PDS_METHOD = PoissonDiskSamplingMethod::FloydSteinbergExpo;

		auto img_data = slimage::ConvertUV(color,
			[&density](unsigned x, unsigned y, const slimage::Pixel3ub& px) {
				return Pixel<PixelRgb>{
					1.0f,
					{
						static_cast<float>(x),
						static_cast<float>(y)
					},
					density(x,y),
					{
						Eigen::Vector3f{
							static_cast<float>(px[0]),
							static_cast<float>(px[1]),
							static_cast<float>(px[2])
						}/255.0f
					}
				};
			});

		auto sp = ALIC(img_data,
			ComputeSeeds(PDS_METHOD, img_data),
			[COMPACTNESS=opt.compactness](const Superpixel<PixelRgb>& a, const Pixel<PixelRgb>& b) {
				return COMPACTNESS * (a.position - b.position).squaredNorm() / (a.radius * a.radius)
					+ (1.0f - COMPACTNESS) * (a.data.color - b.data.color).squaredNorm();
			});

		return sp;
	}


}
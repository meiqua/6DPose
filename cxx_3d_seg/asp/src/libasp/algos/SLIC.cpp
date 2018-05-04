#include <slimage/algorithm.hpp>
#include <asp/algos.hpp>
#include <asp/alic.hpp>

namespace asp
{

	Segmentation<PixelRgb> SuperpixelsSlic(const slimage::Image3ub& img_rgb, const SlicParameters& opt)
	{
		const float density = static_cast<float>(opt.num_superpixels) / (img_rgb.width() * img_rgb.height());
		auto img_data = slimage::ConvertUV(img_rgb,
			[density](unsigned x, unsigned y, const slimage::Pixel3ub& px) {
				return Pixel<PixelRgb>{
					1.0f,
					{
						static_cast<float>(x),
						static_cast<float>(y)
					},
					density,
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
			ComputeSeeds(PoissonDiskSamplingMethod::Grid, img_data),
			[COMPACTNESS=opt.compactness](const Superpixel<PixelRgb>& a, const Pixel<PixelRgb>& b) {
				return COMPACTNESS * (a.position - b.position).squaredNorm() / (a.radius * a.radius)
					+ (1.0f - COMPACTNESS) * (a.data.color - b.data.color).squaredNorm();
			});

		return sp;
	}


}
#include <asp/algos.hpp>
#include <asp/alic.hpp>
#include <slimage/image.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <map>

namespace asp
{
namespace group_helper {
struct Vertex{
    int id;
    int idx;
    int parent;
    int count=1;
    std::vector<size_t> children;
    std::vector<Eigen::Vector3f> worlds;
    std::vector<Eigen::Vector3f> normals;

    bool operator>(const Vertex& rhs) const{
      return count > rhs.count;
    }
};
struct Edge{
    int v1;
    int v2;
    int count = 1;
    float weight = 0;
    bool operator<(const Edge& rhs) const{
      return weight < rhs.weight;
    }
};

int find(int idx, std::vector<Vertex>& vertices){
    int u_parent = vertices[idx].parent;
    while (u_parent!=vertices[u_parent].parent) {
        u_parent = vertices[u_parent].parent;
    }
    return u_parent;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx;
}
}

	/** Computes first derivative for 5 evenly spaced depth samples (v0,...,v4) */
	float LocalFiniteDifferencesPrimesense(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3, uint16_t v4)
	{
		const float v0f = static_cast<float>(v0);
		const float v1f = static_cast<float>(v1);
		const float v2f = static_cast<float>(v2);
		const float v3f = static_cast<float>(v3);
		const float v4f = static_cast<float>(v4);

		if(v0 == 0 && v4 == 0 && v1 != 0 && v3 != 0) {
			return v3f - v1f;
		}

		bool left_invalid = (v0 == 0 || v1 == 0);
		bool right_invalid = (v3 == 0 || v4 == 0);
		if(left_invalid && right_invalid) {
			return 0.0f;
		}
		else if(left_invalid) {
			return v4f - v2f;
		}
		else if(right_invalid) {
			return v2f - v0f;
		}
		else {
			float a = std::abs(v2f + v0f - 2.0f*v1f);
			float b = std::abs(v4f + v2f - 2.0f*v3f);
			float p, q;
			if(a + b == 0.0f) {
				p = 0.5f;
				q = 0.5f;
			}
			else {
				p = a/(a + b);
				q = b/(a + b);
			}
			return q*(v2f - v0f) + p*(v4f - v2f);
		}
	}

	/** Computes depth gradient for pixel (j,i) */
	Eigen::Vector2f LocalDepthGradient(const slimage::Image1ui16& depth, unsigned int j, unsigned int i, const DaspParameters& opt)
	{
		uint16_t d00 = depth(j,i);

		float z_over_f = static_cast<float>(d00) * opt.depth_to_z / opt.focal_px;
		float window = 0.1f * opt.radius / z_over_f;

		// compute w = base_scale*f/d
		unsigned int w = std::max(static_cast<unsigned int>(window + 0.5f), 4u);
		if(w % 2 == 1) w++;

		// can not compute the gradient at the border, so return 0
		if(i < w || depth.height() - w <= i || j < w || depth.width() - w <= j) {
			return Eigen::Vector2f::Zero();
		}

		float dx = LocalFiniteDifferencesPrimesense(
			depth(j-w,i),
			depth(j-w/2,i),
			d00,
			depth(j+w/2,i),
			depth(j+w,i)
		);

		float dy = LocalFiniteDifferencesPrimesense(
			depth(j,i-w),
			depth(j,i-w/2),
			d00,
			depth(j,i+w/2),
			depth(j,i+w)
		);

		// Theoretically scale == base_scale, but w must be an integer, so we
		// compute scale from the actually used w.

		// compute 1 / scale = 1 / (w*d/f)
		float scl = 1.0f / (float(w) * z_over_f);

		return (scl*opt.depth_to_z) * Eigen::Vector2f(static_cast<float>(dx), static_cast<float>(dy));
	}

	/** Computes normal from gradient and assures that it points towards the camera (which is in 0) */
	Eigen::Vector3f NormalFromGradient(const Eigen::Vector2f& g, const Eigen::Vector3f& position)
	{
		const float gx = g.x();
		const float gy = g.y();
		const float scl = 1.0f / std::sqrt(1.0f + gx*gx + gy*gy); // Danvil::MoreMath::FastInverseSqrt
		Eigen::Vector3f normal(scl*gx, scl*gy, -scl);
		// force normal to look towards the camera
		// check if point to camera direction and normal are within 90 deg
		// enforce: normal * (cam_pos - pos) > 0
		// do not need to normalize (cam_pos - pos) as only sign is considered
		const float q = normal.dot(-position);
		if(q < 0) {
			normal *= -1.0f;
		}
		return normal;
	}

	/** Computes 3D point for a pixel */
	Eigen::Vector3f Backproject(const Eigen::Vector2f& pos, const Eigen::Vector2f& center, float depth, const DaspParameters& opt)
	{
		return (depth / opt.focal_px) * Eigen::Vector3f{ pos.x() - center.x(), pos.y() - center.y(), opt.focal_px };
	}

	/** Computes DASP density */
	float Density(float depth, const Eigen::Vector2f& gradient, const DaspParameters& opt)
	{
		float q = depth / (opt.radius * opt.focal_px);
		return q * q / 3.1415f * std::sqrt(gradient.squaredNorm() + 1.0f);
	}

	/** Normal distance function */
	float NormalDistance(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
	{
		// approximation for the angle between the two normals
		return 1.0f - a.dot(b);
	}

	Segmentation<PixelRgbd> SuperpixelsDasp(const slimage::Image3ub& img_rgb, const slimage::Image1ui16& img_d, const DaspParameters& opt_in)
	{
		constexpr PoissonDiskSamplingMethod PDS_METHOD = PoissonDiskSamplingMethod::FloydSteinbergExpo;
		const DaspParameters opt = opt_in; // use local copy for higher performance
		const unsigned width = img_rgb.width();
		const unsigned height = img_d.height();
        const Eigen::Vector2f cam_center = Eigen::Vector2f{ static_cast<float>(opt.cx), static_cast<float>(opt.cy) };

		slimage::Image<Pixel<PixelRgbd>,1> img_data{width, height};
		for(unsigned y=0, i=0; y<height; y++) {
			for(unsigned x=0; x<width; x++, i++) {
				const auto& rgb = img_rgb(x,y);
				auto idepth = img_d(x,y);
				Pixel<PixelRgbd>& q = img_data(x,y);
				q.position = { static_cast<float>(x), static_cast<float>(y) };
				q.data.color = Eigen::Vector3f{ static_cast<float>(rgb[0]), static_cast<float>(rgb[1]), static_cast<float>(rgb[2]) }/255.0f;
				if(idepth == 0) {
					// invalid pixel
					q.num = 0.0f;
					q.data.depth = 0.0f;
					q.data.world = Eigen::Vector3f::Zero();
//					Eigen::Vector2f gradient = Eigen::Vector2f::Zero();
					q.density = 0.0f;
					q.data.normal = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
				}
				else {
					// normal pixel
					q.num = 1.0f;
					q.data.depth = static_cast<float>(idepth) * opt.depth_to_z;
					q.data.world = Backproject(q.position, cam_center, q.data.depth, opt);
					Eigen::Vector2f gradient = LocalDepthGradient(img_d, x, y, opt);
					q.density = Density(q.data.depth, gradient, opt);
					q.data.normal = NormalFromGradient(gradient, q.data.world);
				}
			}
		}

		if(opt.num_superpixels > 0) {
			// compute current total density = number of superpixels
			float total_density = 0.0f;
			for(const auto& q : img_data) {
				total_density += q.density;
			}
			// compute density scale factor
			float density_scale_factor = static_cast<float>(opt.num_superpixels) / total_density;
			// scale density
			for(auto& q : img_data) {
				q.density *= density_scale_factor;
			}
		}

		auto sp = ALIC(img_data,
			ComputeSeeds(PDS_METHOD, img_data),
			[COMPACTNESS=opt.compactness, NORMAL_WEIGHT=opt.normal_weight, RADIUS_SCL=1.0f/(opt.radius*opt.radius)]
			(const asp::Superpixel<PixelRgbd>& a, const Pixel<PixelRgbd>& b) {
				return
					COMPACTNESS * (a.data.world - b.data.world).squaredNorm() * RADIUS_SCL
					+ (1.0f - COMPACTNESS) * (
						(1.0f - NORMAL_WEIGHT) * (a.data.color - b.data.color).squaredNorm()
						+ NORMAL_WEIGHT * NormalDistance(a.data.normal, b.data.normal)
					);
			});

//		std::cout << sp.superpixels.size() << " superpixels" << std::endl;

		return sp;
	}

    slimage::Image<int,1> DsapGrouping(const slimage::Image3ub& img_rgb,
                                       const slimage::Image1ui16& img_d,
                                       const DaspParameters& opt_in,
                                       slimage::Image3f& sli_world,
                                       slimage::Image3f& sli_normal){

        auto seg = SuperpixelsDasp(img_rgb, img_d, opt_in);

        auto R_seed = opt_in.radius;
        auto& indices = seg.indices;
        const auto width = indices.width();
        const auto height = indices.height();

        sli_world.resize(width, height);
        sli_normal.resize(width, height);
        for(int y=0; y<height; y++) {
            for(int x=0; x<width; x++) {
                auto pixel = seg.input(x,y);

                for(int i=0; i<3; i++){
                    sli_world(x,y)[i] = pixel.data.world(i);
                    sli_normal(x,y)[i] = pixel.data.normal(i);
                }
            }
        }

        std::vector<int> unique_id;
        for(int y=0; y<height; y++) {
            for(int x=0; x<width; x++) {
                int i0 = indices(x,y);
                if(i0 == -1) {
                    continue;
                }
                if (std::find(unique_id.begin(), unique_id.end(), i0) == unique_id.end())
                    unique_id.push_back(i0);
            }
        }
        std::sort(unique_id.begin(), unique_id.end());
        std::map<int, int> id2idx;
        for(int i=0; i<unique_id.size();i++){
            id2idx[unique_id[i]] = i;
        }

        std::vector<group_helper::Vertex> vertices;
        std::vector<group_helper::Edge> edges;
        for(size_t i=0;i < unique_id.size();i++){
            group_helper::Vertex v;
            v.id = unique_id[i];
            v.idx = i;
            v.parent = i;
            v.worlds.push_back(seg.superpixels[v.id].data.world);
            v.normals.push_back(seg.superpixels[v.id].data.normal);
            v.count = seg.superpixels[v.id].num;
            v.children.push_back(i);
            vertices.push_back(v);
        }

        slimage::Image<int,1> adj_table{uint32_t(unique_id.size()), uint32_t(unique_id.size())};
        for(int y=0; y<height-1; y++) {
            for(int x=0; x<width-1; x++) {

                int i0 = indices(x,y);
                if(i0 == -1) {
                    continue;
                }
                int i1 = indices(x+1,y);
                int i2 = indices(x,y+1);
                if(i0 != i1 && i1 != -1) {
                    int idx0 = id2idx.find(i0)->second;
                    int idx1 = id2idx.find(i1)->second;
                    adj_table(idx0, idx1) += 1;
                    adj_table(idx1, idx0) += 1;
                }
                if(i0 != i2 && i2 != -1) {
                    int idx0 = id2idx.find(i0)->second;
                    int idx2 = id2idx.find(i2)->second;
                    adj_table(idx0, idx2) += 1;
                    adj_table(idx2, idx0) += 1;
                }
            }
        }


        for(int y=0; y<unique_id.size();y++){
            for(int x=y+1; x<unique_id.size();x++){
                if(adj_table(x,y)>0){
                    group_helper::Edge edge;
                    edge.v1 = unique_id[x];
                    edge.v2 = unique_id[y];
                    edge.count = adj_table(x,y);

                    auto& super1 = seg.superpixels[edge.v1];
                    auto& super2 = seg.superpixels[edge.v2];
                    auto& center1 = super1.data.world;
                    auto& center2 = super2.data.world;
                    auto& normal1 = super1.data.normal;
                    auto& normal2 = super2.data.normal;

                    auto center1_2 = center1-center2;
                    bool isConvex = true;
                    auto convex_angle = center1_2.normalized().dot(normal1);
                    auto convex_angle2 = -center1_2.normalized().dot(normal2);
                    if(convex_angle<-0.2f || convex_angle2<-0.2f){
                        isConvex = false;
                    }
                    float weight =(1-std::abs(normal1.dot(normal2)));
                    if(!isConvex || center1_2.norm()/R_seed>3){
                        continue;  // don't want the edge
                    }
                    edge.weight = weight;
                    edges.push_back(edge);
                }
            }
        }
        std::sort(edges.begin(), edges.end());

        for(auto& edge: edges){
            int v1 = edge.v1;
            int v2 = edge.v2;
            int idx1 = id2idx.find(v1)->second;
            int idx2 = id2idx.find(v2)->second;
            int parent1 = group_helper::find(idx1, vertices);
            int parent2 = group_helper::find(idx2, vertices);
            if(parent1!=parent2){
                // weighted union find
                if(vertices[parent1].count>vertices[parent2].count){
                    int temp= parent1;
                    parent1 = parent2;
                    parent2 = temp;
                }
                if(edge.count>R_seed*400){
                    // plane grouping
                    if(edge.weight<0.02)
                    {
                        vertices[parent1].parent = parent2;

                        vertices[parent2].worlds.insert(vertices[parent2].worlds.end(),
                                                        vertices[parent1].worlds.begin(),
                                                        vertices[parent1].worlds.end());
                        vertices[parent2].normals.insert(vertices[parent2].normals.end(),
                                                        vertices[parent1].normals.begin(),
                                                        vertices[parent1].normals.end());
                        vertices[parent2].children.insert(vertices[parent2].children.end(),
                                                        vertices[parent1].children.begin(),
                                                        vertices[parent1].children.end());
                        vertices[parent2].count += vertices[parent1].count;

                        adj_table(idx1,idx2) = 0;
                        adj_table(idx2,idx1) = 0;
                    }else{
                        break;
                    }
                }
            }
        }

        auto idx_sorted = group_helper::sort_indexes(vertices);
        for(auto parent2: idx_sorted){
            auto& v = vertices[parent2];
            if(v.idx == v.parent){
                // find adj segments
                for(size_t child_idx=0;child_idx<v.children.size();child_idx++){
                    size_t c = v.children[child_idx];
                    for(size_t x=0; x<unique_id.size(); x++){
                        if(c != x){
                            if(adj_table(x, c) > R_seed*400){
                                int parent1 = group_helper::find(x, vertices);
                                if(parent2 != parent1){
                                    int convex_check = 0;
                                    int convex_thresh = 1;
                                    for(int i=0; i<vertices[parent1].worlds.size(); i++){
                                        auto& world1 = vertices[parent1].worlds[i];
                                        auto& normal1 = vertices[parent1].normals[i];
                                        for(int j=0; j<vertices[parent2].worlds.size(); j++){
                                            auto& world2 = vertices[parent2].worlds[j];
                                            auto& normal2 = vertices[parent2].normals[j];

                                            auto center2_1 = (world2-world1).normalized();
                                            auto convex_angle = center2_1.dot(normal2);
                                            auto convex_angle2 = -center2_1.dot(normal1);
                                            if(convex_angle<-0.1f || convex_angle2<-0.1f){
                                                convex_check += 1;
                                                if(convex_check>convex_thresh){
                                                    break;
                                                }
                                            }
                                        }
                                        if(convex_check>convex_thresh){
                                            break;
                                        }
                                    }
                                    if(convex_check<=convex_thresh){
                                        vertices[parent1].parent = parent2;
                                        vertices[parent2].worlds.insert(vertices[parent2].worlds.end(),
                                                                        vertices[parent1].worlds.begin(),
                                                                        vertices[parent1].worlds.end());
                                        vertices[parent2].normals.insert(vertices[parent2].normals.end(),
                                                                        vertices[parent1].normals.begin(),
                                                                        vertices[parent1].normals.end());
                                        vertices[parent2].children.insert(vertices[parent2].children.end(),
                                                                        vertices[parent1].children.begin(),
                                                                        vertices[parent1].children.end());
                                        vertices[parent2].count += vertices[parent1].count;
                                    }
                                }
                                adj_table(x, c) = 0;
                                adj_table(c, x) = 0;
                            }
                        }
                    }
                }
            }
        }


        std::vector<int> segs_v;
        std::vector<int> segs_count;
        std::map<int, int> id2new;
        for(int i=0;i<vertices.size();i++){
            auto& v = vertices[i];
            if(v.parent==v.idx){
                segs_v.push_back(i);
                int count = 0;
                for(auto c: v.children){
                    count += vertices[c].count;
                }
                segs_count.push_back(count);
            }
        }
        auto count_sorted = group_helper::sort_indexes(segs_count);

        auto group = seg.indices;
        for(int i=0; i<count_sorted.size();i++){
            for(auto c: vertices[segs_v[count_sorted[i]]].children){
                id2new[vertices[c].id] = i;
            }
        }

        for(int y=0; y<height; y++) {
            for(int x=0; x<width; x++) {
                int i0 = group(x,y);
                if(i0 == -1) {
                    continue;
                }
                group(x,y) = id2new.find(i0)->second;
            }
        }
        return group;
    }
}
